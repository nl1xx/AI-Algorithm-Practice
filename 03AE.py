from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 20),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(20, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()    # 图片数值取值为[0,1]，不宜用ReLU
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, 784)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(batch_size, 1, 28, 28)
        return x


data_train = datasets.MNIST(root="./datasets", train=True, transform=transforms.Compose([transforms.ToTensor()]),
                            download=True)
data_test = datasets.MNIST(root="./datasets", train=False, transform=transforms.Compose([transforms.ToTensor()]),
                           download=True)

data_train = DataLoader(dataset=data_train, batch_size=32, shuffle=True)
data_test = DataLoader(dataset=data_test, batch_size=32, shuffle=True)

x, label = iter(data_train).__next__()  # 取出第一批(batch)训练所用的数据集
print(x.shape)  # img :  torch.Size([32, 1, 28, 28])， 每次迭代获取32张图片，每张图大小为(1,28,28)

model = AE().to('cuda')
loss_function = nn.MSELoss()
lr = 1e-3
optimizer = optim.Adam(params=model.parameters(), lr=lr)
loss_list = []
epoch = 10
for i in range(epoch):
    for data in data_train:
        x, label = data
        x = x.to('cuda')
        x_hat = model(x)
        optimizer.zero_grad()
        loss = loss_function(x_hat, x)
        loss.backward()
        optimizer.step()
    loss_list.append(loss.item())
    print('epoch: {}, loss: {}'.format(i, loss.item()))
    i += 1

plt.plot(loss_list)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()
