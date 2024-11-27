import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())

x = torch.randn(1)
if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))
    print(torch.version.cuda)

print(torch.backends.cudnn.version())
# 能够正确返回8801
from torch.backends import cudnn  # 若正常则静默
print(cudnn.is_available())
# 若正常返回True
a = torch.tensor(1.)
print(cudnn.is_acceptable(a.cuda()))
# 若正常返回True
