# U-Net + DDPM
# https://blog.csdn.net/Lizhi_Tech/article/details/132174887

import torch, os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image


# U-Net
class ResidualConvBlock(nn.Module):  # DoubleConv
    def __init__(
            self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        '''
        process and downscale the image feature maps
        '''
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        '''
        process and upscale the image feature maps
        '''
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class Unet(nn.Module):
    def __init__(self, in_channels, n_feat=256):
        super(Unet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2 * n_feat)
        self.timeembed2 = EmbedFC(1, 1 * n_feat)

        self.up0 = nn.Sequential(
            # nn.ConvTranspose2d(6 * n_feat, 2 * n_feat, 7, 7), # when concat temb and cemb end up w 6*n_feat
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7),  # otherwise just have 2*n_feat
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, t):
        """
        输入加噪图像和对应的时间step，预测反向噪声的正态分布
        :param x: 加噪图像
        :param t: 对应step
        :return: 正态分布噪声
        """
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        # embed time step
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        # 将上采样输出与step编码相加，输入到下一个上采样层
        up1 = self.up0(hiddenvec)
        up2 = self.up1(up1 + temb1, down2)
        up3 = self.up2(up2 + temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out


# Diffusion model
class DDPM(nn.Module):
    def __init__(self, model, betas, n_T, device):
        super(DDPM, self).__init__()
        self.model = model.to(device)

        # register_buffer 可以提前保存alpha相关，节约时间
        for k, v in self.ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.loss_mse = nn.MSELoss()

    # forward(算法一)
    def ddpm_schedules(self, beta1, beta2, T):
        """
        提前计算各个step的alpha，这里beta是线性变化
        :param beta1: beta的下限
        :param beta2: beta的上限
        :param T: 总共的step数
        """
        assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

        beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1  # 生成beta1-beta2均匀分布的数组
        sqrt_beta_t = torch.sqrt(beta_t)
        alpha_t = 1 - beta_t
        log_alpha_t = torch.log(alpha_t)
        alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()  # alpha累乘

        sqrtab = torch.sqrt(alphabar_t)  # 根号alpha累乘
        oneover_sqrta = 1 / torch.sqrt(alpha_t)  # 1 / 根号alpha

        sqrtmab = torch.sqrt(1 - alphabar_t)  # 根号下（1-alpha累乘）
        mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

        return {
            "alpha_t": alpha_t,  # \alpha_t
            "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
            "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
            "alphabar_t": alphabar_t,  # \bar{\alpha_t}
            "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}} # 加噪标准差
            "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}  # 加噪均值
            "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
        }

    def forward(self, x):
        """
        训练过程中, 随机选择step和生成噪声
        """
        # 随机选择step
        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        # 随机生成正态分布噪声
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        # 加噪后的图像x_t
        x_t = (
                self.sqrtab[_ts, None, None, None] * x
                + self.sqrtmab[_ts, None, None, None] * noise
        )

        # 将unet预测的对应step的正态分布噪声与真实噪声做对比
        # 将加噪后的图像x_t和标准化的时间步_ts/self.n_T作为输入，预测在该时间步对应的噪声
        return self.loss_mse(noise, self.model(x_t, _ts / self.n_T))

    # 从随机的初始噪声开始预测当前噪声的上一个step的正态分布噪声,然后根据采样公式得到反向扩散的均值和方差,最后根据重整化公式计算出上一个step的图像
    # backward(算法二)
    def sample(self, n_sample, size, device):
        # 随机生成初始噪声图片 x_T ~ N(0, 1)
        x_i = torch.randn(n_sample, *size).to(device)
        for i in range(self.n_T, 0, -1):
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample, 1, 1, 1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            eps = self.model(x_i, t_is)
            x_i = x_i[:n_sample]
            x_i = self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i]) + self.sqrt_beta_t[i] * z
        return x_i


class ImageGenerator(object):
    def __init__(self):
        """
        初始化，定义超参数、数据集、网络结构等
        """
        self.epoch = 20
        self.sample_num = 100
        self.batch_size = 64
        self.lr = 0.0001
        self.n_T = 400
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.init_dataloader()
        self.sampler = DDPM(model=Unet(in_channels=1), betas=(1e-4, 0.02), n_T=self.n_T, device=self.device).to(
            self.device)
        self.optimizer = optim.Adam(self.sampler.model.parameters(), lr=self.lr)

    def init_dataloader(self):
        """
        初始化数据集和dataloader
        """
        tf = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = MNIST('./datasets/',
                              train=True,
                              download=True,
                              transform=tf)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        val_dataset = MNIST('./datasets/',
                            train=False,
                            download=True,
                            transform=tf)
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

    def train(self):
        self.sampler.train()
        print('Start Training!!')
        for epoch in range(self.epoch):
            self.sampler.model.train()
            loss_mean = 0
            for i, (images, labels) in enumerate(self.train_dataloader):
                images, labels = images.to(self.device), labels.to(self.device)

                # 将latent和condition拼接后输入网络
                loss = self.sampler(images)
                loss_mean += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            train_loss = loss_mean / len(self.train_dataloader)
            print('epoch:{}, loss:{:.4f}'.format(epoch, train_loss))
            self.visualize_results(epoch)

    @torch.no_grad()
    def visualize_results(self, epoch):
        self.sampler.eval()
        # 保存结果路径
        output_path = './images/DiffusionGeneration'
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        tot_num_samples = self.sample_num
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))
        out = self.sampler.sample(tot_num_samples, (1, 28, 28), self.device)
        save_image(out, os.path.join(output_path, '{}.jpg'.format(epoch)), nrow=image_frame_dim)


if __name__ == '__main__':
    generator = ImageGenerator()
    generator.train()
