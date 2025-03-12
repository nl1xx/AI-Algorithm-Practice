from torch import optim
from torchvision import transforms
from torchvision.models import vgg16
from torch.utils.data import Dataset
from glob import glob
import os
import time
import numpy as np
import torch
import torch.nn as nn
import logging
from optparse import OptionParser
from torch.utils.data import random_split, DataLoader
import visdom
from PIL import Image


# VGG
class VGG(nn.Module):
    def __init__(self, pretrained=False):
        super(VGG, self).__init__()

        # conv1 1/2
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv2 1/4
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv3 1/8
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv4 1/16
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv5 1/32
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 加载VGG16
        if pretrained:
            pretrained_model = vgg16(pretrained=pretrained)
            pretrained_params = pretrained_model.state_dict()
            keys = list(pretrained_params.keys())
            new_dict = {}
            for index, key in enumerate(self.state_dict().keys()):
                new_dict[key] = pretrained_params[keys[index]]
            self.load_state_dict(new_dict)

    def forward(self, x):
        x = self.relu1_1(self.conv1_1(x))
        x = self.relu1_2(self.conv1_2(x))
        x = self.pool1(x)
        pool1 = x

        x = self.relu2_1(self.conv2_1(x))
        x = self.relu2_2(self.conv2_2(x))
        x = self.pool2(x)
        pool2 = x

        x = self.relu3_1(self.conv3_1(x))
        x = self.relu3_2(self.conv3_2(x))
        x = self.relu3_3(self.conv3_3(x))
        x = self.pool3(x)
        pool3 = x

        x = self.relu4_1(self.conv4_1(x))
        x = self.relu4_2(self.conv4_2(x))
        x = self.relu4_3(self.conv4_3(x))
        x = self.pool4(x)
        pool4 = x

        x = self.relu5_1(self.conv5_1(x))
        x = self.relu5_2(self.conv5_2(x))
        x = self.relu5_3(self.conv5_3(x))
        x = self.pool5(x)
        pool5 = x

        return pool1, pool2, pool3, pool4, pool5


# FCN
class FCN32s(nn.Module):
    def __init__(self, num_classes, backbone="vgg"):
        super(FCN32s, self).__init__()
        self.num_classes = num_classes
        if backbone == "vgg":
            self.features = VGG()

        # deConv1 1/16
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu1 = nn.ReLU()

        # deConv1 1/8
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU()

        # deConv1 1/4
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        # deConv1 1/2
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()

        # deConv1 1/1
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.relu5 = nn.ReLU()

        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        features = self.features(x)

        y = self.bn1(self.relu1(self.deconv1(features[4])))

        y = self.bn2(self.relu2(self.deconv2(y)))

        y = self.bn3(self.relu3(self.deconv3(y)))

        y = self.bn4(self.relu4(self.deconv4(y)))

        y = self.bn5(self.relu5(self.deconv5(y)))

        y = self.classifier(y)

        return y


class FCN16s(nn.Module):
    def __init__(self, num_classes, backbone="vgg"):
        super(FCN16s, self).__init__()
        self.num_classes = num_classes
        if backbone == "vgg":
            self.features = VGG()

        # deConv1 1/16
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu1 = nn.ReLU()

        # deConv1 1/8
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU()

        # deConv1 1/4
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        # deConv1 1/2
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()

        # deConv1 1/1
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.relu5 = nn.ReLU()

        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        features = self.features(x)

        y = self.bn1(self.relu1(self.deconv1(features[4])) + features[3])

        y = self.bn2(self.relu2(self.deconv2(y)))

        y = self.bn3(self.relu3(self.deconv3(y)))

        y = self.bn4(self.relu4(self.deconv4(y)))

        y = self.bn5(self.relu5(self.deconv5(y)))

        y = self.classifier(y)

        return y


class FCN8s(nn.Module):
    def __init__(self, num_classes, backbone="vgg"):
        super(FCN8s, self).__init__()
        self.num_classes = num_classes
        if backbone == "vgg":
            self.features = VGG()

        # deConv1 1/16
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu1 = nn.ReLU()

        # deConv1 1/8
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU()

        # deConv1 1/4
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        # deConv1 1/2
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()

        # deConv1 1/1
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.relu5 = nn.ReLU()

        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        features = self.features(x)

        y = self.bn1(self.relu1(self.deconv1(features[4])) + features[3])

        y = self.bn2(self.relu2(self.deconv2(y)) + features[2])

        y = self.bn3(self.relu3(self.deconv3(y)))

        y = self.bn4(self.relu4(self.deconv4(y)))

        y = self.bn5(self.relu5(self.deconv5(y)))

        y = self.classifier(y)

        return y


class FCNs(nn.Module):
    def __init__(self, num_classes, backbone="vgg"):
        super(FCNs, self).__init__()
        self.num_classes = num_classes
        if backbone == "vgg":
            self.features = VGG()

        # deConv1 1/16
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu1 = nn.ReLU()

        # deConv1 1/8
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU()

        # deConv1 1/4
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        # deConv1 1/2
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()

        # deConv1 1/1
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.relu5 = nn.ReLU()

        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        features = self.features(x)

        y = self.bn1(self.relu1(self.deconv1(features[4])) + features[3])

        y = self.bn2(self.relu2(self.deconv2(y)) + features[2])

        y = self.bn3(self.relu3(self.deconv3(y)) + features[1])

        y = self.bn4(self.relu4(self.deconv4(y)) + features[0])

        y = self.bn5(self.relu5(self.deconv5(y)))

        y = self.classifier(y)

        return y


# Dataset
class CustomDataset(Dataset):
    def __init__(self, image_path="./my_datasets/BagImages", mode="train"):
        assert mode in ("train", "val", "test")
        self.image_path = image_path
        self.image_list = glob(os.path.join(self.image_path, "*.jpg"))
        self.mode = mode

        if mode in ("train", "val"):
            self.mask_path = self.image_path + "Masks"

        self.transform_x = transforms.Compose([transforms.Resize((256, 256)),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                               ])
        self.transform_mask = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        if self.mode in ("train", "val"):
            image_name = self.image_list[index].split("\\")[-1].split(".")[0]
            X = Image.open(self.image_list[index])

            # convert('1')是一个图像模式转换方法, 用于将图像转换为二值图像(黑白图像)
            mask = np.array(Image.open(os.path.join(self.mask_path, image_name + ".jpg")).convert('1').resize((256, 256)))
            masks = np.zeros((mask.shape[0], mask.shape[1], 2), dtype=np.uint8)
            masks[:, :, 0] = mask
            masks[:, :, 1] = ~mask

            X = self.transform_x(X)
            masks = self.transform_mask(masks) * 255
            return X, masks
        else:
            X = Image.open(self.image_list[index])
            X = self.transform_x(X)
            path = self.image_list[index]
            return X, path

    def __len__(self):
        return len(self.image_list)


CUDA = torch.cuda.is_available()


# train
def train(model, criterion, data_loader, optimizer, epoch, save_freq, save_dir, verbose, device):
    start_time = time.time()
    print(f"Epoch {epoch + 1}, Learning Rate: {optimizer.param_groups[0]['lr']}")
    model.train()

    epoch_loss = 0.0
    batches = 0
    for i, (image, target) in enumerate(data_loader):
        image, target = image.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        batches += 1

        if (i + 1) % verbose == 0:
            print(f"Training Loss: {epoch_loss / batches:.6f}")

    # 保存模型
    if epoch % save_freq == 0:
        state_dict = model.state_dict()
        torch.save({
            'epoch': epoch,
            'state_dict': state_dict,
        }, os.path.join(save_dir, f"{epoch + 1:03d}.ckpt"))
        print(f"Checkpoint saved to {os.path.join(save_dir, f'{epoch + 1:03d}.ckpt')}")

    end_time = time.time()
    print(f"Epoch {epoch + 1} completed. Loss: {epoch_loss / batches:.6f}, Time: {end_time - start_time:.2f} s")


# val
def validate(model, criterion, data_loader, verbose, device):
    start_time = time.time()
    model.eval()

    epoch_loss = 0.0
    with torch.no_grad():
        for i, (image, target) in enumerate(data_loader):
            image, target = image.to(device), target.to(device)
            output = model(image)
            loss = criterion(output, target)

            epoch_loss += loss.item()

            if (i + 1) % verbose == 0:
                print(f"Validation Loss: {epoch_loss / (i + 1):.6f}")

    end_time = time.time()
    print(f"Validation completed. Loss: {epoch_loss / len(data_loader):.6f}, Time: {end_time - start_time:.2f} s")


# test
def test(model, data_loader, device):
    start_time = time.time()
    model.eval()

    for i, (image, path) in enumerate(data_loader):
        image = image.to(device)
        with torch.no_grad():
            output = model(image)

        pred = output.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)

        for j, p in enumerate(path):
            im = Image.fromarray(pred.astype('uint8')[j] * 255, "L")
            save_path = os.path.join("data/testPreds", os.path.basename(p))
            im.save(save_path)

    end_time = time.time()
    print(f"Testing completed. Time: {end_time - start_time:.2f} s")


def main():
    # 参数设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 80
    batch_size = 16
    lr = 1e-2
    save_freq = 1
    save_dir = "./models"
    verbose = 100
    num_classes = 2
    backbone = "vgg"
    mode = "train"
    ckpt_path = None  # 如果需要加载模型，设置为模型路径

    # 初始化模型
    model = FCNs(num_classes, backbone)
    model.to(device)

    # 加载模型（如果有）
    if ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        print(f"Model loaded from {ckpt_path}")

    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 数据集和数据加载器
    train_dataset = CustomDataset(mode="train")
    val_dataset = CustomDataset(mode="val")
    test_dataset = CustomDataset(mode="test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # 训练或测试
    if mode == "train":
        for epoch in range(epochs):
            train(model, criterion, train_loader, optimizer, epoch, save_freq, save_dir, verbose, device)
            validate(model, criterion, val_loader, verbose, device)
    elif mode == "test":
        test(model, test_loader, device)


if __name__ == '__main__':
    main()
