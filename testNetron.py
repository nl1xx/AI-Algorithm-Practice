# pip install netron
# pip install onnx

import torch
from torchvision.models import AlexNet
import netron

model = AlexNet()
input = torch.ones((1, 3, 224, 224))
torch.onnx.export(model, input, f='AlexNet.onnx')  # 导出 .onnx文件
netron.start('AlexNet.onnx')  # 展示结构图
