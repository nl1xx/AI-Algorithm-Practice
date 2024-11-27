# 选择合适的线程数(多线程)
# 经测试num_workers为6最高效

from time import time
import multiprocessing as mp
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])

train_set = torchvision.datasets.MNIST(root="./datasets", train=True, transform=transform, download=True)


if __name__ == '__main__':
    print(f"num of CPU: {mp.cpu_count()}")
    for num_workers in range(2, mp.cpu_count(), 2):
        train_loader = DataLoader(train_set, shuffle=True, num_workers=num_workers, batch_size=64, pin_memory=True)
        start = time()
        for epoch in range(1, 3):
            for i, data in enumerate(train_loader, 0):
                pass
        end = time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))
