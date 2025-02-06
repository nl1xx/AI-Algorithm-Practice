import os
from skimage import util, io, feature, color, segmentation
import numpy as np
import pandas as pd
import cv2 as cv
import shutil
from multiprocessing import Process, Lock
import threading
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
import random
from torch.utils.data import Dataset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import warnings
# 抑制警告
warnings.filterwarnings("ignore")


# utils
def progress_bar(total: int, finished: int, length: int = 50):
    """
    进度条
    :param total: 任务总数
    :param finished: 已完成数量
    :param length: 进度条长度
    :return: None
    """
    percent = finished / total
    arrow = "-" * int(percent * length) + ">"
    spaces = "▓" * (length - len(arrow))
    end = "\n" if finished == total else ""
    print("\r进度: {0}% [{1}] {2}|{3}".format(int(percent * 100), arrow + spaces, finished, total), end=end)
    return

def cal_IoU(boxes: np.ndarray, gt_box) -> np.ndarray:
    """
    计算推荐区域与真值的IoU
    :param boxes: 推荐区域边界框, n*4维数组, 列对应左上和右下两个点坐标[x1, y1, w, h]
    :param gt_box: 真值, 对应左上和右下两个点坐标[x1, y1, w, h]
    :return: iou, 推荐区域boxes与真值的IoU结果
    """
    # 复制矩阵防止直接引用修改原始值
    bbox = boxes.copy()
    gt = gt_box.copy()

    # 将宽度和高度转换成坐标
    bbox[:, 2] = bbox[:, 0] + bbox[:, 2]
    bbox[:, 3] = bbox[:, 1] + bbox[:, 3]
    gt[2] = gt[0] + gt[2]
    gt[3] = gt[1] + gt[3]

    box_area = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
    gt_area = (gt[2] - gt[0]) * (gt[3] - gt[1])

    # 取右下角的最小值减去左上角的最大值，得到交集的宽度和高度
    inter_w = np.minimum(bbox[:, 2], gt[2]) - np.maximum(bbox[:, 0], gt[0])
    inter_h = np.minimum(bbox[:, 3], gt[3]) - np.maximum(bbox[:, 1], gt[1])

    inter = np.maximum(inter_w, 0) * np.maximum(inter_h, 0)
    union = box_area + gt_area - inter
    iou = inter / union
    return iou

def cal_norm_params(root):
    """
    计算数据集归一化参数
    :param root: 待计算数据文件路径
    :return: 数据集的RGB分量均值和标准差
    """
    # 计算RGB分量均值
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((227, 227))])
    data = ImageFolder(root=root, transform=transform)
    m_r, m_g, m_b, s_r, s_g, s_b, = 0, 0, 0, 0, 0, 0
    print('正在计算数据集RGB分量均值和标准差...')
    for idx, info in enumerate(data):
        img = info[0]
        avg = torch.mean(img, dim=(1, 2))
        std = torch.std(img, dim=(1, 2))
        m_r += avg[0].item()
        m_g += avg[1].item()
        m_b += avg[2].item()
        s_r += std[0].item()
        s_g += std[1].item()
        s_b += std[2].item()
        progress_bar(total=len(data), finished=idx + 1)

    m_r = round(m_r / idx, 3)
    m_g = round(m_g / idx, 3)
    m_b = round(m_b / idx, 3)
    s_r = round(s_r / idx, 3)
    s_g = round(s_g / idx, 3)
    s_b = round(s_b / idx, 3)
    norm_params = [m_r, m_g, m_b, s_r, s_g, s_b]
    return norm_params

def Alexnet(pretrained=True, num_classes=2):
    """
    获取AlexNet模型结构
    :param pretrained: 是否加载预训练参数
    :param num_classes: 目标类别数
    :return: AlexNet
    """
    net = models.alexnet(pretrained=pretrained)
    net.classifier[-1] = nn.Linear(in_features=4096, out_features=num_classes)
    return net

def show_predict(dataset, network, device, transform=None, save: str = None):
    """
    显示微调分类模型结果
    :param dataset: 数据集
    :param network: 模型结构
    :param device: CPU/GPU
    :param transform: 数据预处理方法
    :param save: str-保存图像文件的名称, None-不保存
    :return:
    """
    network.eval()
    network.to(device)

    plt.figure(figsize=(30, 30))
    for i in range(12):
        im_path, label = random.choice(dataset.flist)
        name = im_path.split(os.sep)[-1]
        img = io.imread(im_path)
        if transform is not None:
            in_tensor = transform(img).unsqueeze(0).to(device)
        else:
            in_tensor = torch.tensor(img).unsqueeze(0).to(device)

        output = network(in_tensor)
        predict = torch.argmax(output)
        plt.subplot(2, 6, i + 1)
        plt.imshow(img)
        plt.title("{name}\ntruth:{label}\npredict:{predict}".format(name=name, label=label, predict=predict))
    plt.tight_layout()
    if save is not None:
        plt.savefig("./model/predict_" + save + ".jpg")
    plt.show()

def draw_box(img, boxes=None, save_name: str = None):
    """
    在图像上绘制边界框
    :param img: 输入图像
    :param boxes: bbox坐标, 列分别为[x, y, w, h]
    :param save_name: 保存bbox图像名称, None-不保存
    :return: None
    """
    plt.imshow(img)
    axis = plt.gca()
    if boxes is not None:
        for box in boxes:
            rect = patches.Rectangle((int(box[0]), int(box[1])), int(box[2]), int(box[3]), linewidth=1, edgecolor='r', facecolor='none')
            axis.add_patch(rect)
    if save_name is not None:
        os.makedirs("./predict", exist_ok=True)
        plt.savefig("./predict/" + save_name + ".jpg")
    plt.show()
    return None

class RegressDataSet(Dataset):
    def __init__(self, ss_csv_path, gt_csv_path, network, device, transform=None):
        """
        生成回归数据集
        :param ss_csv_path: 存储ss-bbox的文件路径
        :param gt_csv_path: 存储gt-bbox的文件路径
        :param network: 特征提取网络
        :param device: CPU/GPU
        :param transform: 数据预处理方法
        """
        self.ss_csv = pd.read_csv(ss_csv_path, header=None, index_col=None)
        self.gt_csv = pd.read_csv(gt_csv_path, header=0, index_col=None)
        self.gt_csv = self.rename()
        self.network = network
        self.device = device
        self.transform = transform

    def rename(self):
        """
        重命名gt_csv的name对象
        :return: gt_csv
        """
        for idx in range(self.gt_csv.shape[0]):
            fullname = self.gt_csv.iat[idx, 0]
            name = fullname.split("/")[-1]
            self.gt_csv.iat[idx, 0] = name
        return self.gt_csv

    def __getitem__(self, index):
        ss_img_path, *ss_loc = self.ss_csv.iloc[index, :5]
        target_name = ss_img_path.split(os.sep)[-1].rsplit("_", 1)[0] + ".jpg"
        gt_loc = self.gt_csv[self.gt_csv.name == target_name].iloc[0, 2: 6].tolist()
        label = torch.tensor(gt_loc, dtype=torch.float32) - torch.tensor(ss_loc, dtype=torch.float32)

        ss_img = io.imread(ss_img_path)
        ss_img = self.transform(ss_img).to(self.device).unsqueeze(0)
        ss_features = self.network.features(ss_img).squeeze(0)
        return ss_features, label

    def __len__(self):
        return len(self.ss_csv)


class DataSet(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.flist = self.get_flist()

    def get_flist(self):
        """
        获取数据路径及标签列表
        :return: flist-数据路径和对应标签列表
        """
        flist = []
        for roots, dirs, files in os.walk(self.root):
            for file in files:
                if not file.endswith(".jpg"):
                    continue
                im_path = os.path.join(roots, file)
                im_label = int(im_path.split(os.sep)[-2])
                flist.append([im_path, im_label])
        return flist

    def __getitem__(self, index):
        path, label = self.flist[index]
        img = io.imread(path)

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.flist)
# ----------------------------------------------------------------------------------------------------------------------


# "Selective Search for Object Recognition" by J.R.R. Uijlings et al.
def _generate_segments(im_orig, scale, sigma, min_size):
    """
    根据Felzenswalb-Huttenlocher方法将图像分割为小区域图像
    :param im_orig: 输入3通道图像
    :param scale: 分割参数, 数值越小, 分割越精细
    :param sigma: 分割图像前对图像进行高斯平滑的参数
    :param min_size: 分割的最小单元, 一般设置10-100间
    :return: 带分割类别的4通道图
    """
    # 获取分割后每个小区域所属的类别
    im_mask = segmentation.felzenszwalb(util.img_as_float(im_orig), scale=scale, sigma=sigma, min_size=min_size)
    # 把类别合并到最后一个通道上, 维度为[w, h, 4]
    im_orig = np.append(im_orig, np.zeros(im_orig.shape[:2])[:, :, np.newaxis], axis=2)
    im_orig[:, :, 3] = im_mask
    return im_orig


def _sim_colour(r1, r2):
    """
    计算区域颜色直方图交集和
    :param r1: 区域 1
    :param r2:区域 2
    :return: 颜色直方图交集和
    """
    return sum([min(a, b) for a, b in zip(r1["hist_c"], r2["hist_c"])])


def _sim_texture(r1, r2):
    """
    计算区域纹理直方图交集和
    :param r1: 区域 1
    :param r2:区域 2
    :return: 颜色直方图交集和
    """
    return sum([min(a, b) for a, b in zip(r1["hist_t"], r2["hist_t"])])


def _sim_size(r1, r2, imsize):
    """
    计算图像大小相似度
    :param r1: 区域 1
    :param r2: 区域 2
    :param imsize: 图像size
    :return: 图像大小相似度
    """
    return 1.0 - (r1["size"] + r2["size"]) / imsize


def _sim_fill(r1, r2, imsize):
    """
    计算图像填充相似度
    :param r1: 区域 1
    :param r2: 区域 2
    :param imsize: 图像带下
    :return: 填充相似度结果
    """
    bbsize = (
        (max(r1["max_x"], r2["max_x"]) - min(r1["min_x"], r2["min_x"]))
        * (max(r1["max_y"], r2["max_y"]) - min(r1["min_y"], r2["min_y"]))
    )
    return 1.0 - (bbsize - r1["size"] - r2["size"]) / imsize


def _calc_sim(r1, r2, imsize):
    """
    整合区域相似度结果
    :param r1: 区域 1
    :param r2: 区域 2
    :param imsize: 整体相似度结果
    :return:
    """
    return _sim_colour(r1, r2) + _sim_texture(r1, r2) + _sim_size(r1, r2, imsize) + _sim_fill(r1, r2, imsize)


def _calc_colour_hist(img):
    """
    在HSV空间计算图像颜色直方图, 输出维度为[BINS, COLOUR_CHANNELS(3)], 参考[uijlings_ijcv2013_draft.pdf]这里bins取值为25
    :param img: hsv空间图像
    :return: 颜色直方图
    """
    BINS = 25
    hist = np.array([])

    for colour_channel in (0, 1, 2):
        # extracting one colour channel
        c = img[:, colour_channel]

        # calculate histogram for each colour and join to the result
        hist = np.concatenate([hist] + [np.histogram(c, BINS, (0.0, 255.0))[0]])

    # L1 normalize
    hist = hist / len(img)
    return hist


def _calc_texture_gradient(img):
    """
    计算纹理梯度, 原始ss方法采用Gaussian导数方法, 此处采用lbp方法
    :param img: 输入图像
    :return: 和输入图像等大的lbp纹理特征
    """
    ret = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    for colour_channel in (0, 1, 2):
        ret[:, :, colour_channel] = feature.local_binary_pattern(img[:, :, colour_channel], 8, 1.0)
    return ret


def _calc_texture_hist(img):
    """
    计算图像每个通道的纹理直方图
    :param img: 输入图像
    :return: 纹理直方图
    """
    BINS = 10
    hist = np.array([])

    for colour_channel in (0, 1, 2):
        # mask by the colour channel
        fd = img[:, colour_channel]
        # calculate histogram for each orientation and concatenate them all
        # and join to the result
        hist = np.concatenate([hist] + [np.histogram(fd, BINS, (0.0, 1.0))[0]])

    # L1 Normalize
    hist = hist / len(img)
    return hist


# Efficient Graph-Based Image Segmentation
def _extract_regions(img):
    """
    提取原始图像分割区域
    :param img: 带像素标签的4通道图像
    :return: 原始分割区域
    """
    R = {}
    # get hsv image
    hsv = color.rgb2hsv(img[:, :, :3])

    # step-1 像素位置计数
    # 遍历每个像素标签, 若其还没有被分配到一个区域, 就创建一个新的区域并将其添加到字典R中
    for y, i in enumerate(img):
        for x, (r, g, b, l) in enumerate(i):
            # initialize a new region
            if l not in R:
                R[l] = {
                    "min_x": 0xffff, "min_y": 0xffff,
                    "max_x": 0, "max_y": 0, "labels": [l]}

            # bounding box
            # 根据该像素的坐标更新该区域的边界框, 即最小和最大的x和y坐标
            if R[l]["min_x"] > x:
                R[l]["min_x"] = x
            if R[l]["min_y"] > y:
                R[l]["min_y"] = y
            if R[l]["max_x"] < x:
                R[l]["max_x"] = x
            if R[l]["max_y"] < y:
                R[l]["max_y"] = y

    # step-2 计算纹理梯度
    tex_grad = _calc_texture_gradient(img)

    # step-3 计算颜色和纹理直方图
    for k, v in R.items():
        # colour histogram
        masked_pixels = hsv[:, :, :][img[:, :, 3] == k]
        R[k]["size"] = len(masked_pixels / 4)
        R[k]["hist_c"] = _calc_colour_hist(masked_pixels)

        # texture histogram
        R[k]["hist_t"] = _calc_texture_hist(tex_grad[:, :][img[:, :, 3] == k])
    return R


def _extract_neighbours(regions):
    """
    提取给定区域之间的相邻关系
    :param regions: 输入所有区域
    :return: list-所有相交的区域对
    """
    def intersect(a, b):
        """
        判断两个区域是否相交
        :param a: 区域 a
        :param b: 区域 b
        :return: bool-区域是否相交
        """
        if (a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
            a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]):
            return True
        return False

    R = regions.items()
    r = [elm for elm in R]
    R = r
    neighbours = []
    for cur, a in enumerate(R[:-1]):
        for b in R[cur + 1:]:
            if intersect(a[1], b[1]):
                neighbours.append((a, b))
    return neighbours


def _merge_regions(r1, r2):
    """
    区域合并并更新颜色直方图/纹理直方图/大小
    :param r1: 区域 1
    :param r2: 区域 2
    :return: 合并后的区域
    """
    new_size = r1["size"] + r2["size"]
    rt = {
        "min_x": min(r1["min_x"], r2["min_x"]),
        "min_y": min(r1["min_y"], r2["min_y"]),
        "max_x": max(r1["max_x"], r2["max_x"]),
        "max_y": max(r1["max_y"], r2["max_y"]),
        "size": new_size,
        "hist_c": (
            r1["hist_c"] * r1["size"] + r2["hist_c"] * r2["size"]) / new_size,
        "hist_t": (
            r1["hist_t"] * r1["size"] + r2["hist_t"] * r2["size"]) / new_size,
        "labels": r1["labels"] + r2["labels"]
    }
    return rt


# 按照SS的算法执行Step 3
def selective_search(im_orig, scale=1.0, sigma=0.8, min_size=50):
    """
    选择性搜索生成候选区域
    :param im_orig: 输入3通道图像
    :param scale: 分割参数, 数值越小, 分割越精细
    :param sigma: 分割图像前对图像进行高斯平滑的参数
    :param min_size: 分割的最小单元, 一般设置10-100间
    :return: img-带有区域标签的图像(r, g, b, region), regions-字典{”rect“:(left, top, width, height), "labels":[...]}
    """
    assert im_orig.shape[2] == 3, "3ch image is expected"

    # 加载图像获取最小分割区域
    # 区域标签存储在每个像素的第四个通道 [r, g, b, region]
    img = _generate_segments(im_orig, scale, sigma, min_size)

    if img is None:
        return None, {}

    imsize = img.shape[0] * img.shape[1]
    R = _extract_regions(img)

    # 获取相邻区域对
    neighbours = _extract_neighbours(R)

    # 计算初始相似度
    S = {}
    for (ai, ar), (bi, br) in neighbours:
        S[(ai, bi)] = _calc_sim(ar, br, imsize)

    # 进行层次搜索, 直到没有新的相似度可以计算
    while S != {}:

        # 获取两最大相似度区域的下标(i, j)
        i, j = sorted(list(S.items()), key=lambda a: a[1])[-1][0]

        # 将最大相似度区域合并为一个新的区域rt
        t = max(R.keys()) + 1.0
        R[t] = _merge_regions(R[i], R[j])

        # 标记相似度集合中与(i, j)相关的区域, 并将其移除
        key_to_delete = []
        for k, v in S.items():
            if (i in k) or (j in k):
                key_to_delete.append(k)

        # 移除相关区域
        for k in key_to_delete:
            del S[k]

        # 计算与新区域rt与相邻区域的相似度并添加到集合S中
        for k in filter(lambda a: a != (i, j), key_to_delete):
            n = k[1] if k[0] in (i, j) else k[0]
            S[(t, n)] = _calc_sim(R[t], R[n], imsize)

    regions = []
    for k, r in R.items():
        regions.append({
            'rect': (r['min_x'], r['min_y'], r['max_x'] - r['min_x'], r['max_y'] - r['min_y']),
            'size': r['size'],
            'labels': r['labels']
        })
    return img, regions
# ----------------------------------------------------------------------------------------------------------------------


class SelectiveSearch:
    def __init__(self, root, max_pos_regions: int = None, max_neg_regions: int = None, threshold=0.5):
        """
        采用ss方法生成候选区域文件
        :param root: 训练/验证数据集所在路径
        :param max_pos_regions: 每张图片最多产生的正样本候选区域个数, None表示不进行限制
        :param max_neg_regions: 每张图片最多产生的负样本候选区域个数, None表示不进行限制
        :param threshold: IoU进行正负样本区分时的阈值
        """
        self.source_root = os.path.join(root, 'source')
        self.ss_root = os.path.join(root, 'ss')
        self.csv_path = os.path.join(self.source_root, "gt_loc.csv")
        self.max_pos_regions = max_pos_regions
        self.max_neg_regions = max_neg_regions
        self.threshold = threshold
        self.info = None

    @staticmethod
    def cal_proposals(img, scale=200, sigma=0.7, min_size=20, use_cv=True) -> np.ndarray:
        """
        计算后续区域坐标
        :param img: 原始输入图像
        :param scale: 控制ss方法初始聚类大小
        :param sigma: ss方法高斯核参数
        :param min_size: ss方法最小像素数
        :param use_cv: (bool) true-采用cv生成候选区域, false-利用源码生成
        :return: candidates, 候选区域坐标矩阵n*4维, 每列分别对应[x, y, w, h]
        """
        rows, cols, channels = img.shape
        if use_cv:
            # 生成候选区域
            ss = cv.ximgproc.segmentation.createSelectiveSearchSegmentation()
            ss.setBaseImage(img)
            ss.switchToSelectiveSearchFast()
            proposals = ss.process()
            candidates = set()
            # 对区域进行限制
            for region in proposals:
                rect = tuple(region)
                if rect in candidates:
                    continue
                # # 限制区域框形状和大小
                # x1, y1, w, h = rect
                # if w * h < 500:
                #     continue
                # if w / h > 2 or h / w > 2 or w / cols < 0.05 or h / rows < 0.05:
                #     continue
                candidates.add(rect)
        else:
            # ss方法返回4通道图像img_lbl, 其前三通道为rgb值, 最后一个通道表示该proposal-region在ss方法实现过程中所属的区域标签
            # ss方法返回字典regions, regions['rect']为(x, y, w, h), regions['size']为像素数,  regions['labels']为区域包含的对象的类别标签
            img_lbl, regions = selective_search(im_orig=img, scale=scale, sigma=sigma, min_size=min_size)
            candidates = set()
            for region in regions:
                # excluding same rectangle with different segments
                if region['rect'] in candidates:
                    continue
                # # 限制区域框形状和大小
                # x1, y1, w, h = rect
                # if w * h < 500:
                #     continue
                # if w / h > 2 or h / w > 2 or w / cols < 0.05 or h / rows < 0.05:
                #     continue
                candidates.add(region['rect'])
        candidates = np.array(list(candidates))
        return candidates

    def save(self, num_workers=1, method="thread"):
        """
        生成目标区域并保存
        :param num_workers: 进程或线程数
        :param method: 多进程-process或者多线程-thread
        :return: None
        """
        self.info = pd.read_csv(self.csv_path, header=0, index_col=None)
        # label为0存储背景图, label不为0存储带目标图像
        categories = list(self.info['label'].unique())
        categories.append(0)
        for category in categories:
            folder = os.path.join(self.ss_root, str(category))
            os.makedirs(folder, exist_ok=True)
        index = self.info.index.to_list()
        span = len(index) // num_workers
        # 使用文件锁进行后续文件写入, 防止多进程或多线程由于并发写入出现的竞态条件, 即多个线程或进程同时访问和修改同一资源时，导致数据不一致或出现意外的结果
        # 获取文件锁，确保只有一个进程或线程可以执行写入操作。在完成写入操作后，释放文件锁，允许其他进程或线程进行写入。防止过程中出现错误或者空行等情况
        lock = Lock()
        # 多进程生成图像
        if "process" in method.lower():
            print("=" * 8 + "开始多进程生成候选区域图像" + "=" * 8)
            processes = []
            for i in range(num_workers):
                if i != num_workers - 1:
                    p = Process(target=self.save_proposals, kwargs={'lock': lock, 'index': index[i * span: (i + 1) * span]})
                else:
                    p = Process(target=self.save_proposals, kwargs={'lock': lock, 'index': index[i * span:]})
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
        # 多线程生成图像
        elif "thread" in method.lower():
            print("=" * 8 + "开始多线程生成候选区域图像" + "=" * 8)
            threads = []
            for i in range(num_workers):
                if i != num_workers - 1:
                    thread = threading.Thread(target=self.save_proposals, kwargs={'lock': lock, 'index': index[i * span: (i + 1) * span]})
                else:
                    thread = threading.Thread(target=self.save_proposals, kwargs={'lock': lock, 'index': index[i * span: (i + 1) * span]})
                thread.start()
                threads.append(thread)
            for thread in threads:
                thread.join()
        else:
            print("=" * 8 + "开始生成候选区域图像" + "=" * 8)
            self.save_proposals(lock=lock, index=index)
        return None

    def save_proposals(self, lock, index, show_fig=False):
        """
        生成候选区域图片并保存相关信息
        :param lock: 文件锁, 防止写入文件错误
        :param index: 文件index
        :param show_fig: 是否展示后续区域划分结果
        :return: None
        """
        for row in index:
            name = self.info.iloc[row, 0]
            label = self.info.iloc[row, 1]
            # gt值为[x, y, w, h]
            gt_box = self.info.iloc[row, 2:].values
            im_path = os.path.join(self.source_root, name)
            img = io.imread(im_path)
            # 计算推荐区域坐标矩阵[x, y, w, h]
            proposals = self.cal_proposals(img=img)

            # 计算proposals与gt的IoU结果
            IoU = cal_IoU(proposals, gt_box)
            # 根据IoU阈值将proposals图像划分到正负样本集
            boxes_p = proposals[np.where(IoU >= self.threshold)]
            boxes_n = proposals[np.where((IoU < self.threshold) & (IoU > 0.1))]

            # 展示proposals结果
            if show_fig:
                fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
                ax.imshow(img)
                for (x, y, w, h) in boxes_p:
                    rect = patches.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=1)
                    ax.add_patch(rect)
                for (x, y, w, h) in boxes_n:
                    rect = patches.Rectangle((x, y), w, h, fill=False, edgecolor='green', linewidth=1)
                    ax.add_patch(rect)
                plt.show()

            # loc.csv用于存储带有目标图像的boxes_p边界框信息
            loc_path = os.path.join(self.ss_root, "ss_loc.csv")

            # 将正样本按照对应label存储到相应文件夹下, 并记录bbox的信息到loc.csv中用于后续bbox回归训练
            num_p = num_n = 0
            for loc in boxes_p:
                num_p += 1
                crop_img = img[loc[1]: loc[1] + loc[3], loc[0]: loc[0] + loc[2], :]
                crop_name = name.split("/")[-1].replace(".jpg", "_" + str(num_p) + ".jpg")
                crop_path = os.path.join(self.ss_root, str(label), crop_name)
                with lock:
                    # 保存的ss区域仍然为[x, y, w, h]
                    with open(loc_path, 'a', newline='') as fa:
                        fa.writelines([crop_path, ',', str(loc[0]), ',', str(loc[1]), ',', str(loc[2]), ',', str(loc[3]), '\n'])
                    fa.close()
                io.imsave(fname=crop_path, arr=crop_img, check_contrast=False)
                if self.max_pos_regions is None:
                    continue
                if num_p == self.max_pos_regions:
                    break

            # 将负样本按照存储到"./0/"文件夹下, 其bbox信息对于回归训练无用, 故不用记录
            for loc in boxes_n:
                num_n += 1
                crop_img = img[loc[1]: loc[1] + loc[3], loc[0]: loc[0] + loc[2], :]
                crop_name = name.split("/")[-1].replace(".jpg", "_" + str(num_n) + ".jpg")
                crop_path = os.path.join(self.ss_root, "0", crop_name)
                io.imsave(fname=crop_path, arr=crop_img, check_contrast=False)
                if self.max_neg_regions is None:
                    continue
                if num_n == self.max_neg_regions:
                    break
            print("{name}: {num_p}个正样本, {num_n}个负样本".format(name=name, num_p=num_p, num_n=num_n))
# ----------------------------------------------------------------------------------------------------------------------


# train
def train(data_loader, network, num_epochs, optimizer, scheduler, criterion, device, train_rate=0.8, mode="classify"):
    """
    模型训练
    :param data_loader: 数据dataloader
    :param network: 网络结构
    :param num_epochs: 训练轮次
    :param optimizer: 优化器
    :param scheduler: 学习率调度器
    :param criterion: 损失函数
    :param device: CPU/GPU
    :param train_rate: 训练集比例
    :param mode: 模型类型, 预训练-pretrain, 分类-classify, 回归-regression
    :return: None
    """
    os.makedirs('./model', exist_ok=True)
    network = network.to(device)
    criterion = criterion.to(device)
    best_acc = 0.0
    best_loss = np.inf
    print("=" * 8 + "开始训练{mode}模型".format(mode=mode.lower()) + "=" * 8)
    batch_num = len(data_loader)
    train_batch_num = round(batch_num * train_rate)
    train_loss_all, val_loss_all, train_acc_all, val_acc_all = [], [], [], []

    for epoch in range(num_epochs):
        train_num = val_num = 0
        train_loss = val_loss = 0.0
        train_corrects = val_corrects = 0
        for step, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)
            # 模型训练
            if step < train_batch_num:
                network.train()
                y_hat = network(x)
                loss = criterion(y_hat, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 计算每个batch的loss结果与预测正确的数量
                label_hat = torch.argmax(y_hat, dim=1)
                # 预训练/分类模型计算loss和acc, 回归模型只计算loss
                if mode.lower() == 'pretrain' or mode.lower() == 'classify':
                    train_corrects += (label_hat == y).sum().item()
                train_loss += loss.item() * x.size(0)
                train_num += x.size(0)
            # 模型验证
            else:
                network.eval()
                with torch.no_grad():
                    y_hat = network(x)
                    loss = criterion(y_hat, y)
                    label_hat = torch.argmax(y_hat, dim=1)
                    if mode.lower() == 'pretrain' or mode.lower() == 'classify':
                        val_corrects += (label_hat == y).sum().item()
                    val_loss += loss.item() * x.size(0)
                    val_num += x.size(0)

        scheduler.step()
        # 记录loss和acc变化曲线
        train_loss_all.append(train_loss / train_num)
        val_loss_all.append(val_loss / val_num)
        if mode.lower() == 'pretrain' or mode.lower() == 'classify':
            train_acc_all.append(100 * train_corrects / train_num)
            val_acc_all.append(100 * val_corrects / val_num)
            print("Mode:{}  Epoch:[{:0>3}|{}]  train_loss:{:.3f}  train_acc:{:.2f}%  val_loss:{:.3f}  val_acc:{:.2f}%".format(
                mode.lower(), epoch + 1, num_epochs,
                train_loss_all[-1], train_acc_all[-1],
                val_loss_all[-1], val_acc_all[-1]
            ))
        else:
            print("Mode:{}  Epoch:[{:0>3}|{}]  train_loss:{:.3f}  val_loss:{:.3f}".format(
                mode.lower(), epoch + 1, num_epochs,
                train_loss_all[-1], val_loss_all[-1]
            ))

        # 保存模型
        # 预训练/分类模型选取准确率最高的参数
        if mode.lower() == "pretrain" or mode.lower() == "classify":
            if val_acc_all[-1] > best_acc:
                best_acc = val_acc_all[-1]
                save_path = os.path.join("./model", mode + ".pth")
                # torch.save(network.state_dict(), save_path)
                torch.save(network, save_path)
        # 回归模型选取损失最低的参数
        else:
            if val_loss_all[-1] < best_loss:
                best_loss = val_loss_all[-1]
                save_path = os.path.join("./model", mode + ".pth")
                # torch.save(network.state_dict(), save_path)
                torch.save(network, save_path)

    # 绘制训练曲线
    if mode.lower() == "pretrain" or mode.lower() == "classify":
        fig_path = os.path.join("./model/", mode + "_curve.png")
        plt.subplot(121)
        plt.plot(range(num_epochs), train_loss_all, "r-", label="train")
        plt.plot(range(num_epochs), val_loss_all, "b-", label="val")
        plt.title("Loss")
        plt.legend()
        plt.subplot(122)
        plt.plot(range(num_epochs), train_acc_all, "r-", label="train")
        plt.plot(range(num_epochs), val_acc_all, "b-", label="val")
        plt.title("Acc")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
    else:
        fig_path = os.path.join("./model/", mode + "_curve.png")
        plt.plot(range(num_epochs), train_loss_all, "r-", label="train")
        plt.plot(range(num_epochs), val_loss_all, "b-", label="val")
        plt.title("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
    return None


def run(train_root=None, network=None, batch_size=64, criterion=None, device=None, train_rate=0.8,
        epochs=10, lr=0.001, mode="classify", show_fig=False):
    """
    模型训练
    :param train_root: 待训练数据路径
    :param network: 模型结构
    :param batch_size: batch size
    :param criterion: 损失函数
    :param device: CPU/GPU
    :param train_rate: 训练集比率
    :param epochs: 训练轮次
    :param lr: 学习率
    :param mode: 模型类型
    :param show_fig: 是否展示训练结果
    :return: None
    """

    # 判断transform参数文件是否存在
    transform_params_path = "./model/pretrain_transform_params.csv" if mode == "pretrain" else "./model/classify_transform_params.csv"
    exist = os.path.exists(transform_params_path)
    if not exist:
        print("正在计算{}模型归一化参数...".format(mode))
        transform_params = cal_norm_params(root=train_root)
        pf = pd.DataFrame(transform_params)
        pf.to_csv(transform_params_path, header=False, index=False)
    else:
        transform_params = pd.read_csv(transform_params_path, header=None, index_col=None).values
        transform_params = [x[0] for x in transform_params]

    # transforms数据预处理
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((227, 227)),
                                    transforms.Normalize(mean=transform_params[0: 3], std=transform_params[3: 6])])

    # 判断模型是否已经存在
    model_path = "./model/" + mode + ".pth"
    exist = os.path.exists(model_path)
    if not exist:
        print("目标路径下不存在{}模型".format(mode))

        # 预训练和分类模型直接加载数据文件
        if mode == "pretrain" or mode == "classify":
            optimizer = torch.optim.SGD(params=network.parameters(), lr=lr, momentum=0.9, weight_decay=0.1)
            scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
            train_set = DataSet(root=train_root, transform=transform)
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            # 模型训练
            train(data_loader=train_loader, network=network, num_epochs=epochs, optimizer=optimizer, scheduler=scheduler,
                  criterion=criterion, device=device, train_rate=train_rate, mode=mode)
        # 回归模型需利用分类模型计算特征, 作为模型输入
        else:
            # 加载分类模型
            classifier = torch.load("./model/classify.pth")
            # 加载回归任务数据文件
            ss_csv_path = "./my_datasets/data/ss/ss_loc.csv"
            gt_csv_path = "./my_datasets/data/source/gt_loc.csv"
            print("正在利用微调分类模型计算特征作为回归模型的输入...")
            train_set = RegressDataSet(ss_csv_path=ss_csv_path, gt_csv_path=gt_csv_path, network=classifier,
                                             device=device, transform=transform)
            train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
            print("已完成回归模型数据集创建")

            # 定义线性回归模型并初始化权重
            regressor = nn.Sequential(nn.AdaptiveAvgPool2d((6, 6)), nn.Flatten(), nn.Linear(256 * 6 * 6, 4))
            nn.init.xavier_normal_(regressor[-1].weight)
            optimizer = torch.optim.SGD(params=regressor.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
            scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
            # 训练回归模型
            train(data_loader=train_loader, network=regressor, num_epochs=epochs, optimizer=optimizer, scheduler=scheduler,
                  criterion=criterion, device=device, train_rate=0.8, mode="regress")

        # 图像显示训练结果
        if show_fig:
            if mode != "regress":
                show_predict(dataset=train_set, network=network, device=device, transform=transform, save=mode)
    else:
        print("目标路径下已经存在{}模型".format(mode))
        if show_fig:
            network = torch.load(model_path)
            # 加载数据文件
            train_set = DataSet(root=train_root, transform=transform)
            if mode != "regress":
                show_predict(dataset=train_set, network=network, device=device, transform=transform, save=mode)
    return
# ----------------------------------------------------------------------------------------------------------------------


# predict
def predict(im_path, classifier, regressor, transform, device):
    """
    回归模型预测
    :param im_path: 输入图像路径
    :param classifier: 分类模型
    :param regressor: 回归模型
    :param transform: 预处理方法
    :param device: CPU/GPU
    :return: None
    """
    classifier = classifier.to(device)
    regressor = regressor.to(device)
    # 计算proposal region
    img = io.imread(im_path)
    save_name = im_path.split(os.sep)[-1]
    proposals = SelectiveSearch.cal_proposals(img=img)

    boxes, offsets = [], []
    for box in proposals:
        with torch.no_grad():
            crop = img[box[1]: box[1] + box[3], box[0]: box[0] + box[2], :]
            crop_tensor = transform(crop).unsqueeze(0).to(device)
            # 分类模型检测有物体, 才进行后续回归模型计算坐标偏移值
            out = classifier(crop_tensor)
            if torch.argmax(out).item():
                features = classifier.features(crop_tensor)
                offset = regressor(features).squeeze(0).to(device)
                offsets.append(offset)
                boxes.append(torch.tensor(box, dtype=torch.float32, device=device))

    if boxes is not None:
        offsets, boxes = torch.vstack(offsets), torch.vstack(boxes)
        # 以坐标偏移的L1范数最小作为最终box选择标准
        index = offsets.abs().sum(dim=1).argmin().item()
        boxes = boxes[index] + offsets[index]
        draw_box(img, np.array(boxes.unsqueeze(0).cpu()), save_name=save_name)
    else:
        draw_box(img, save_name=save_name)
    return None


if __name__ == "__main__":
    data_root = "./my_datasets/data"
    ss_root = os.path.join(data_root, "ss")
    if os.path.exists(ss_root):
        print("正在删除{}目录下原有数据".format(ss_root))
        shutil.rmtree(ss_root)
    print("正在利用选择性搜索方法创建数据集: {}".format(ss_root))
    select = SelectiveSearch(root=data_root, max_pos_regions=None, max_neg_regions=40, threshold=0.5)
    select.save(num_workers=os.cpu_count(), method="thread")
    # ------------------------------------------------------------------------------------------------------------------

    if not os.path.exists("./my_datasets/data/ss"):
        raise FileNotFoundError("数据不存在, 请先运行SelectiveSearch.py生成目标区域")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = [nn.CrossEntropyLoss(), nn.MSELoss()]
    model_root = "./model"
    os.makedirs(model_root, exist_ok=True)

    # 在17flowers数据集上进行预训练
    pretrain_root = "./my_datasets/data/source/17flowers/jpg"
    pretrain_net = Alexnet(pretrained=True, num_classes=17)
    run(train_root=pretrain_root, network=pretrain_net, batch_size=128, criterion=criterion[0], device=device,
        train_rate=0.8, epochs=15, lr=0.001, mode="pretrain", show_fig=True)

    # 在由2flowers生成的ss数据上进行背景/物体多分类训练
    classify_root = "./my_datasets/data/ss"
    classify_net = torch.load("./model/pretrain.pth")
    classify_net.classifier[-1] = nn.Linear(in_features=4096, out_features=3)
    run(train_root=classify_root, network=classify_net, batch_size=128, criterion=criterion[0], device=device,
        train_rate=0.8, epochs=15, lr=0.001, mode="classify", show_fig=True)

    # 在由2flowers生成的ss物体数据进行边界框回归训练
    run(batch_size=128, criterion=criterion[1], device=device, train_rate=0.8, epochs=50, lr=0.0001, mode="regress", show_fig=False)
    # ------------------------------------------------------------------------------------------------------------------

    device = torch.device('cuda:0')
    # 加载分类模型和回归模型
    classifier_path = './model/classify.pth'
    classifier = torch.load(classifier_path)
    regressor_path = './model/regress.pth'
    regressor = torch.load(regressor_path)
    classifier.eval()
    regressor.eval()

    # transforms数据预处理
    transform_params_path = "./model/classify_transform_params.csv"
    transform_params = pd.read_csv(transform_params_path, header=None, index_col=None).values
    transform_params = [x[0] for x in transform_params]
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((227, 227)),
                                    transforms.Normalize(mean=transform_params[0: 3], std=transform_params[3: 6])])

    root = "./my_datasets/data/source/17flowers"
    for roots, dirs, files in os.walk(root):
        for file in files:
            if not file.endswith(".jpg"):
                continue
            img_path = os.path.join(roots, file)
            predict(im_path=img_path, classifier=classifier, regressor=regressor, transform=transform, device=device)
