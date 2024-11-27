# 1.多尺度输入网络
# 1.1 MTCNN
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


# P-Net(候选网络)
class PNet(nn.Module):

    def __init__(self):
        super(PNet, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(10, 16, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.PReLU()
        )

        self.conv4_1 = nn.Conv2d(32, 1, kernel_size=1, stride=1)
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1, stride=1)
        self.conv4_3 = nn.Conv2d(32, 10, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv_layer(x)

        cond = F.sigmoid(self.conv4_1(x))
        box_offset = self.conv4_2(x)
        land_offset = self.conv4_3(x)

        return cond, box_offset, land_offset


# R-Net(精炼网络)
class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 28, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(28, 48, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(48, 64, kernel_size=2, stride=1),
            nn.PReLU()

        )
        self.line1 = nn.Sequential(
            nn.Linear(64 * 3 * 3, 128),
            nn.PReLU()
        )

        self.line2_1 = nn.Linear(128, 1)
        self.line2_2 = nn.Linear(128, 4)
        self.line2_3 = nn.Linear(128, 10)

    def forward(self, x):

        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.line1(x)

        label = F.sigmoid(self.line2_1(x))
        box_offset = self.line2_2(x)
        land_offset = self.line2_3(x)

        return label, box_offset, land_offset


# O-Net(输出网络)
class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=2, stride=1),
            nn.PReLU()
        )
        self.line1 = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.PReLU()
        )

        self.line2_1 = nn.Linear(256, 1)
        self.line2_2 = nn.Linear(256, 4)
        self.line2_3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.conv5(x)
        x = self.prelu5(x)

        label = F.sigmoid(self.line2_1(x))
        box_offset = self.line2_2(x)
        land_offset = self.line2_3(x)

        return label, box_offset, land_offset


# 2.多尺度特征融合网络
# 2.1 SPP
class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = nn.Conv2d(c1, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv2 = nn.Conv2d(c_ * (len(k) + 1), c2, 1, 1)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


# 2.2 ASPP
class ASPP(nn.Module):  # deeplab

    def __init__(self, dim,in_dim):
        super(ASPP, self).__init__()
        self.down_conv = nn.Sequential(nn.Conv2d(dim,in_dim , 3,padding=1),nn.BatchNorm2d(in_dim),
             nn.PReLU())
        down_dim = in_dim // 2

        self.conv1 = nn.Sequential(nn.Conv2d(in_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=2, padding=2), nn.BatchNorm2d(down_dim), nn.PReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.PReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.PReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(in_dim, down_dim, kernel_size=1),nn.BatchNorm2d(down_dim),  nn.PReLU())
        self.fuse = nn.Sequential(nn.Conv2d(5 * down_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU())

    def forward(self, x):
        x = self.down_conv(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        conv5 = F.upsample(self.conv5(F.adaptive_avg_pool2d(x, 1)), size=x.size()[2:], mode='bilinear')
        return self.fuse(torch.cat((conv1, conv2, conv3,conv4, conv5), 1))


# 2.3 PPM
class PPM(nn.Module): # pspnet
    def __init__(self, down_dim):
        super(PPM, self).__init__()
        self.down_conv = nn.Sequential(nn.Conv2d(2048,down_dim , 3,padding=1),nn.BatchNorm2d(down_dim),
             nn.PReLU())

        self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)),nn.Conv2d(down_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU())
        self.conv2 = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(2, 2)), nn.Conv2d(down_dim, down_dim, kernel_size=1),
            nn.BatchNorm2d(down_dim), nn.PReLU())
        self.conv3 = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(3, 3)),nn.Conv2d(down_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU())
        self.conv4 = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(6, 6)), nn.Conv2d(down_dim, down_dim, kernel_size=1),
            nn.BatchNorm2d(down_dim), nn.PReLU())

        self.fuse = nn.Sequential(nn.Conv2d(4 * down_dim, down_dim, kernel_size=1), nn.BatchNorm2d(down_dim), nn.PReLU())

    def forward(self, x):
        x = self.down_conv(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        conv1_up = F.upsample(conv1, size=x.size()[2:], mode='bilinear')
        conv2_up = F.upsample(conv2, size=x.size()[2:], mode='bilinear')
        conv3_up = F.upsample(conv3, size=x.size()[2:], mode='bilinear')
        conv4_up = F.upsample(conv4, size=x.size()[2:], mode='bilinear')

        return self.fuse(torch.cat((conv1_up, conv2_up, conv3_up, conv4_up), 1))


# 2.4 FPN
# ResNet基本的Bottleneck类(Resnet50/101/152)
class Bottleneck(nn.Module):
    expansion = 4  # 通道扩增倍数(Resnet网络的结构)

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.expansion * planes),
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x  # 初始的x
        out = self.bottleneck(x)
        # 残差融合前需保证out与identity的通道数以及图像尺寸均相同
        if self.downsample is not None:
            identity = self.downsample(x)  # 初始的x采取下采样
        out += identity
        out = self.relu(out)
        return out


class FPN(nn.Module):
    '''
    FPN需要初始化一个list，代表ResNet每一个阶段的Bottleneck的数量
    '''

    def __init__(self, layers):
        super(FPN, self).__init__()
        # 构建C1
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 自下而上搭建C2、C3、C4、C5
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], 2)  # c2->c3第一个bottleneck的stride=2
        self.layer3 = self._make_layer(256, layers[2], 2)  # c3->c4第一个bottleneck的stride=2
        self.layer4 = self._make_layer(512, layers[3], 2)  # c4->c5第一个bottleneck的stride=2

        # 对C5减少通道，得到P5
        self.toplayer = nn.Conv2d(2048, 256, 1, 1, 0)  # 1*1卷积

        # 横向连接，保证每一层通道数一致
        self.latlayer1 = nn.Conv2d(1024, 256, 1, 1, 0)
        self.latlayer2 = nn.Conv2d(512, 256, 1, 1, 0)
        self.latlayer3 = nn.Conv2d(256, 256, 1, 1, 0)

        # 平滑处理 3*3卷积
        self.smooth = nn.Conv2d(256, 256, 3, 1, 1)

    # 构建C2到C5
    def _make_layer(self, planes, blocks, stride=1, downsample=None):
        # 残差连接前，需保证尺寸及通道数相同
        if stride != 1 or self.inplanes != Bottleneck.expansion * planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, Bottleneck.expansion * planes, 1, stride, bias=False),
                nn.BatchNorm2d(Bottleneck.expansion * planes)
            )
        layers = []
        layers.append(Bottleneck(self.inplanes, planes, stride, downsample))

        # 更新输入输出层
        self.inplanes = planes * Bottleneck.expansion

        # 根据block数量添加bottleneck的数量
        for i in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))  # 后面层stride=1
        return nn.Sequential(*layers)  # nn.Sequential接收orderdict或者一系列模型，列表需*转化

        # 自上而下的上采样

    def _upsample_add(self, x, y):
        _, _, H, W = y.shape  # b c h w
        # 特征x 2倍上采样(上采样到y的尺寸)后与y相加
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        # 自下而上
        c1 = self.relu(self.bn1(self.conv1(x)))  # 1/2
        # c1=self.maxpool(self.relu(self.bn1(self.conv1(x))))  此时为1/4
        c2 = self.layer1(self.maxpool(c1))  # 1/4
        c3 = self.layer2(c2)  # 1/8
        c4 = self.layer3(c3)  # 1/16
        c5 = self.layer4(c4)  # 1/32

        # 自上而下，横向连接
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))

        # 平滑处理
        p5 = p5  # p5直接输出
        p4 = self.smooth(p4)
        p3 = self.smooth(p3)
        p2 = self.smooth(p2)
        return p2, p3, p4, p5


FPN(layers=[5, 5, 5, 5])
# 传入生成c2 c3 c4 c5特征层的bottleneck的堆叠数量，返回p2 p3 p4 p5


# 2.5 PANet


# 2.6 U-Net
# 上采样+拼接
class Up(nn.Module):
    def __init__(self,in_channels,out_channels,bilinear=True):
        '''
        :param in_channels: 输入通道数
        :param out_channels:  输出通道数
        :param bilinear: 是否采用双线性插值，默认采用
        '''
        super(Up, self).__init__()
        if bilinear:
            # 双线性差值
            self.up = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
            self.conv = doubleConv(in_channels,out_channels,in_channels//2) # 拼接后为1024，经历第一个卷积后512
        else:
            # 转置卷积实现上采样
            # 输出通道数减半，宽高增加一倍
            self.up = nn.ConvTranspose2d(in_channels,out_channels//2,kernel_size=2,stride=2)
            self.conv = doubleConv(in_channels,out_channels)

    def forward(self,x1,x2):
        # 上采样
        x1 = self.up(x1)
        # 拼接
        x = torch.cat([x1,x2],dim=1)
        # 经历双卷积
        x = self.conv(x)
        return x


# 双卷积层
def doubleConv(in_channels,out_channels,mid_channels=None):
    '''
    :param in_channels: 输入通道数
    :param out_channels: 双卷积后输出的通道数
    :param mid_channels: 中间的通道数，这个主要针对的是最后一个下采样和上采样层
    :return:
    '''
    if mid_channels is None:
        mid_channels = out_channels
    layer = []
    layer.append(nn.Conv2d(in_channels,mid_channels,kernel_size=3,padding=1,bias=False))
    layer.append(nn.BatchNorm2d(mid_channels))
    layer.append(nn.ReLU(inplace=True))
    layer.append(nn.Conv2d(mid_channels,out_channels,kernel_size=3,padding=1,bias=False))
    layer.append(nn.BatchNorm2d(out_channels))
    layer.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layer)


# 下采样
def down(in_channels,out_channels):
    # 池化 + 双卷积
    layer = []
    layer.append(nn.MaxPool2d(2,stride=2))
    layer.append(doubleConv(in_channels,out_channels))
    return nn.Sequential(*layer)


# 整个网络架构
class U_net(nn.Module):
    def __init__(self,in_channels,out_channels,bilinear=True,base_channel=64):
        '''
        :param in_channels: 输入通道数，一般为3，即彩色图像
        :param out_channels: 输出通道数，即网络最后输出的通道数，一般为2，即进行2分类
        :param bilinear: 是否采用双线性插值来上采样，这里默认采取
        :param base_channel: 第一个卷积后的通道数，即64
        '''
        super(U_net, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        # 输入
        self.in_conv = doubleConv(self.in_channels,base_channel)
        # 下采样
        self.down1 = down(base_channel,base_channel*2) # 64,128
        self.down2 = down(base_channel*2,base_channel*4) # 128,256
        self.down3 = down(base_channel*4,base_channel*8) # 256,512
        # 最后一个下采样，通道数不翻倍（因为双线性差值，不会改变通道数的，为了可以简单拼接，就不改变通道数）
        # 当然，是否采取双线新差值，还是由我们自己决定
        factor = 2  if self.bilinear else 1
        self.down4 = down(base_channel*8,base_channel*16 // factor) # 512,512
        # 上采样 + 拼接
        self.up1 = Up(base_channel*16 ,base_channel*8 // factor,self.bilinear) # 1024(双卷积的输入),256（双卷积的输出）
        self.up2 = Up(base_channel*8 ,base_channel*4 // factor,self.bilinear)
        self.up3 = Up(base_channel*4 ,base_channel*2 // factor,self.bilinear)
        self.up4 = Up(base_channel*2 ,base_channel,self.bilinear)
        # 输出
        self.out = nn.Conv2d(in_channels=base_channel,out_channels=self.out_channels,kernel_size=1)

    def forward(self,x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # 不要忘记拼接
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.out(x)

        return {'out':out}


# 3.多尺度特征输出网络
# 3.1 SSD
# https://github.com/amdegroot/ssd.pytorch
