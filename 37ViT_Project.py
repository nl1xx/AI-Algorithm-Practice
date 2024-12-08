# ViT+CIFAR10

import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# MLP Block
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# Attention
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)  # LayerNorm
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        score = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # Transpose k for matmul
        att = self.attend(score)
        att = self.dropout(att)

        out = torch.matmul(att, v)  # Attention output
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)  # Final projection


# Transformer
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, mlp in self.layers:
            x = x + attn(x)
            x = x + mlp(x)
        return self.norm(x)

# ViT
class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        x = self.embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)

# 准备数据集
train_change = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
test_change = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.CIFAR10(root="./xiaotudui/torchvisionDataset", train=True, transform=train_change, download=True)
test_dataset = datasets.CIFAR10(root="./xiaotudui/torchvisionDataset", train=False, transform=test_change, download=True)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)

# 训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViT(224, 16, 10, 256, 2, 4, 128).to(device)
loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
epoch = 10

def train():
    print("Start Training")
    model.train()
    for i in range(epoch):
        total_loss = 0
        for data in train_dataloader:
            optimizer.zero_grad()
            img, label = data
            img = img.to(device)
            label = label.to(device)
            out_img = model(img)
            loss = loss_fn(out_img, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (i + 1) % 2 == 0:
            print("Epoch {}: loss {:.3f}".format(i + 1, total_loss/len(train_dataset)))

# 测试
def test():
    print("Start Test")
    model.eval()
    accuracy_item = 0
    with torch.no_grad():
        for data in test_dataloader:
            img, label = data
            img = img.to(device)
            label = label.to(device)
            predict = model(img)
            accuracy_item += (predict.argmax(1) == label).sum().item()
        print("Accuracy: {:.3f}".format(accuracy_item / len(test_dataset)))

if __name__ == '__main__':
    train()
    test()
