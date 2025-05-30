import torch
import torch.nn as nn


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        # 由572*572*1变成了570*570*64
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.relu1_1 = nn.ReLU(inplace=True)
        # 由570*570*64变成了568*568*64
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.relu1_2 = nn.ReLU(inplace=True)

        # 采用最大池化进行下采样，图片大小减半，通道数不变，由568*568*64变成284*284*64
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 284*284*64->282*282*128
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.relu2_1 = nn.ReLU(inplace=True)
        # 282*282*128->280*280*128
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.relu2_2 = nn.ReLU(inplace=True)

        # 采用最大池化进行下采样  280*280*128->140*140*128
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 140*140*128->138*138*256
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0)
        self.relu3_1 = nn.ReLU(inplace=True)
        # 138*138*256->136*136*256
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.relu3_2 = nn.ReLU(inplace=True)

        # 采用最大池化进行下采样  136*136*256->68*68*256
        self.maxpool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 68*68*256->66*66*512
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=0)
        self.relu4_1 = nn.ReLU(inplace=True)
        # 66*66*512->64*64*512
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0)
        self.relu4_2 = nn.ReLU(inplace=True)

        # 采用最大池化进行下采样  64*64*512->32*32*512
        self.maxpool_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 32*32*512->30*30*1024
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=0)
        self.relu5_1 = nn.ReLU(inplace=True)
        # 30*30*1024->28*28*1024
        self.conv5_2 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=0)
        self.relu5_2 = nn.ReLU(inplace=True)

        # 接下来实现上采样中的up-conv2*2
        # 28*28*1024->56*56*512
        self.up_conv_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2, padding=0)

        # 56*56*1024->54*54*512
        self.conv6_1 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=0)
        self.relu6_1 = nn.ReLU(inplace=True)
        # 54*54*512->52*52*512
        self.conv6_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0)
        self.relu6_2 = nn.ReLU(inplace=True)

        # 52*52*512->104*104*256
        self.up_conv_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=0)

        # 104*104*512->102*102*256
        self.conv7_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=0)
        self.relu7_1 = nn.ReLU(inplace=True)
        # 102*102*256->100*100*256
        self.conv7_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.relu7_2 = nn.ReLU(inplace=True)

        # 100*100*256->200*200*128
        self.up_conv_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0)

        # 200*200*256->198*198*128
        self.conv8_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=0)
        self.relu8_1 = nn.ReLU(inplace=True)
        # 198*198*128->196*196*128
        self.conv8_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.relu8_2 = nn.ReLU(inplace=True)

        # 196*196*128->392*392*64
        self.up_conv_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0)

        # 392*392*128->390*390*64
        self.conv9_1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.relu9_1 = nn.ReLU(inplace=True)
        # 390*390*64->388*388*64
        self.conv9_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.relu9_2 = nn.ReLU(inplace=True)

        # 最后的conv1*1
        self.conv_10 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride=1, padding=0)

    # 中心裁剪
    def crop_tensor(self, tensor, target_tensor):
        target_size = target_tensor.size()[2]
        tensor_size = tensor.size()[2]
        delta = tensor_size - target_size
        delta = delta // 2
        # 如果原始张量的尺寸为10，而delta为2，那么"delta:tensor_size - delta"将截取从索引2到索引8的部分，长度为6，以使得截取后的张量尺寸变为6
        return tensor[:, :, delta:tensor_size - delta, delta:tensor_size - delta]

    def forward(self, x):
        x1 = self.conv1_1(x)
        x1 = self.relu1_1(x1)
        x2 = self.conv1_2(x1)
        x2 = self.relu1_2(x2)  # 这个后续需要使用
        down1 = self.maxpool_1(x2)

        x3 = self.conv2_1(down1)
        x3 = self.relu2_1(x3)
        x4 = self.conv2_2(x3)
        x4 = self.relu2_2(x4)  # 这个后续需要使用
        down2 = self.maxpool_2(x4)

        x5 = self.conv3_1(down2)
        x5 = self.relu3_1(x5)
        x6 = self.conv3_2(x5)
        x6 = self.relu3_2(x6)  # 这个后续需要使用
        down3 = self.maxpool_3(x6)

        x7 = self.conv4_1(down3)
        x7 = self.relu4_1(x7)
        x8 = self.conv4_2(x7)
        x8 = self.relu4_2(x8)  # 这个后续需要使用
        down4 = self.maxpool_4(x8)

        x9 = self.conv5_1(down4)
        x9 = self.relu5_1(x9)
        x10 = self.conv5_2(x9)
        x10 = self.relu5_2(x10)

        # 第一次上采样，需要"Copy and crop"（复制并裁剪）
        up1 = self.up_conv_1(x10)  # 得到56*56*512
        # 需要对x8进行裁剪，从中心往外裁剪
        crop1 = self.crop_tensor(x8, up1)
        up_1 = torch.cat([crop1, up1], dim=1)

        y1 = self.conv6_1(up_1)
        y1 = self.relu6_1(y1)
        y2 = self.conv6_2(y1)
        y2 = self.relu6_2(y2)

        # 第二次上采样，需要"Copy and crop"（复制并裁剪）
        up2 = self.up_conv_2(y2)
        # 需要对x6进行裁剪，从中心往外裁剪
        crop2 = self.crop_tensor(x6, up2)
        up_2 = torch.cat([crop2, up2], dim=1)

        y3 = self.conv7_1(up_2)
        y3 = self.relu7_1(y3)
        y4 = self.conv7_2(y3)
        y4 = self.relu7_2(y4)

        # 第三次上采样，需要"Copy and crop"（复制并裁剪）
        up3 = self.up_conv_3(y4)
        # 需要对x4进行裁剪，从中心往外裁剪
        crop3 = self.crop_tensor(x4, up3)
        up_3 = torch.cat([crop3, up3], dim=1)

        y5 = self.conv8_1(up_3)
        y5 = self.relu8_1(y5)
        y6 = self.conv8_2(y5)
        y6 = self.relu8_2(y6)

        # 第四次上采样，需要"Copy and crop"（复制并裁剪）
        up4 = self.up_conv_4(y6)
        # 需要对x2进行裁剪，从中心往外裁剪
        crop4 = self.crop_tensor(x2, up4)
        up_4 = torch.cat([crop4, up4], dim=1)

        y7 = self.conv9_1(up_4)
        y7 = self.relu9_1(y7)
        y8 = self.conv9_2(y7)
        y8 = self.relu9_2(y8)

        # 最后的conv1*1
        out = self.conv_10(y8)
        return out


if __name__ == '__main__':
    input_data = torch.randn([1, 1, 572, 572])
    unet = Unet()
    output = unet(input_data)
    print(output.shape)
    # torch.Size([1, 2, 388, 388])
