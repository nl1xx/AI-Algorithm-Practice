import torch
import os,glob
import random
import csv
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader


class Dataset_self(Dataset):
    # 第一步：初始化
    def __init__(self, root, mode, resize):
        super(Dataset_self, self).__init__()
        self.resize = resize
        self.root = root
        # 创建一个字典来保存每个文件的标签
        self.name_label = {}
        # 首先得到标签相对于的字典（标签和名称一一对应）
        for name in sorted(os.listdir(os.path.join(root))):
            # 不是文件夹就不需要读取
            if not os.path.isdir(os.path.join(root, name)):
                continue
            # 为每个子目录分配一个唯一的整数标签, 从0开始
            self.name_label[name] = len(self.name_label.keys())
        # print(self.name_label)
        self.image, self.label = self.make_csv('images.csv')
        # 在得到image和label的基础上对图片数据进行一共划分  （注意：如果需要交叉验证就不需要验证集，只划分为训练集和测试集）
        if mode == 'train':
            self.image, self.label = self.image[:int(0.6*len(self.image))], self.label[:int(0.6*len(self.label))]
        if mode == 'val':
            self.image, self.label = (self.image[int(0.6*len(self.image)):int(0.8*len(self.image))],
                                      self.label[int(0.6*len(self.label)):int(0.8*len(self.label))])
        if mode == 'test':
            self.image, self.label = self.image[int(0.8*len(self.image)):], self.label[int(0.8*len(self.label)):]

    # 获得图片和标签的函数
    def make_csv(self, filename):
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for image in self.name_label.keys():
                # 添加图片, 加*贪婪搜索所有关于jpg的文件
                images += glob.glob(os.path.join(self.root, image, '*jpg'))
            # print('长度为：{}，第二张图片为：{}'.format(len(images),images[1]))
            random.shuffle(images)
            # images[0]: ./data\ants\382971067_0bfd33afe0.jpg
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:  # 创建文件
                writer = csv.writer(f)
                for image in images:
                    # 从image中提取出倒数第二级目录的名称(ants)
                    name = image.split(os.sep)[-2]  # 得到与图片相对应的标签
                    label = self.name_label[name]
                    writer.writerow([image, label])  # 写入文件(图片 类别)
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:   # 读取文件
            reader = csv.reader(f)
            for row in reader:
                image, label = row
                label = int(label)
                images.append(image)
                labels.append(label)
        assert len(images) == len(labels)
        return images, labels

    def __len__(self):
        return len(self.image)

    def __getitem__(self, item):   # 单张返回张量的图像与标签
        image, label = self.image[item], self.label[item]
        image = Image.open(image).convert('RGB')
        transf = transforms.Compose([transforms.Resize((int(self.resize), int(self.resize))),
                                     transforms.RandomRotation(15),
                                     transforms.CenterCrop(self.resize),
                                     transforms.ToTensor(),
                                     ])
        image = transf(image)
        label = torch.tensor(label)
        return image, label


def data_dataloader(data_path, mode, size, batch_size, num_workers):
    dataset = Dataset_self(data_path, mode, size)
    dataloader = DataLoader(dataset, batch_size, num_workers)
    return dataloader


def main():
    test = Dataset_self('./data', 'train', 64)


if __name__ == '__main__':
    main()
