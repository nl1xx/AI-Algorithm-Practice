# https://github.com/devJWSong/transformer-translator-pytorch
# Transformer实现英译中

import pandas as pd
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from torch import nn, optim
import math


# Step 1: 数据预处理

# 读取原始数据并提取英文和中文句子
file_path = 'datasets/transformerDataset/cmn.txt'

# 读取文件并处理每一行，提取英文和中文句子
data = []
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            english_sentence = parts[0].strip()
            chinese_sentence = parts[1].strip()
            data.append([english_sentence, chinese_sentence])

# 创建DataFrame保存提取的句子
df = pd.DataFrame(data, columns=['English', 'Chinese'])

df['English'].to_csv('datasets/transformerDataset/english_sentences.txt', index=False, header=False)
df['Chinese'].to_csv('datasets/transformerDataset/chinese_sentences.txt', index=False, header=False)

# 显示前五行数据
print(df.head())


# Step 2: 数据加载与分词

# 英文分词器
tokenizer_en = get_tokenizer('basic_english')


# 中文分词器：将每个汉字作为一个token
def tokenizer_zh(text):
    return list(text)


# 构建词汇表的函数
def build_vocab(sentences, tokenizer):
    """
    根据给定的句子列表和分词器构建词汇表。
    :param sentences: 句子列表
    :param tokenizer: 分词器函数
    :return: 词汇表对象
    """
    def yield_tokens(sentences):
        for sentence in sentences:
            yield tokenizer(sentence)
    vocab = build_vocab_from_iterator(yield_tokens(sentences), specials=['<unk>', '<pad>', '<bos>', '<eos>'])
    vocab.set_default_index(vocab['<unk>'])  # 设置默认索引为 <unk>
    return vocab


# 从文件中加载句子
with open('datasets/transformerDataset/english_sentences.txt', 'r', encoding='utf-8') as f:
    english_sentences = [line.strip() for line in f]

with open('datasets/transformerDataset/chinese_sentences.txt', 'r', encoding='utf-8') as f:
    chinese_sentences = [line.strip() for line in f]

# 构建英文和中文的词汇表
en_vocab = build_vocab(english_sentences, tokenizer_en)
zh_vocab = build_vocab(chinese_sentences, tokenizer_zh)
print(f'英文词汇表大小：{len(en_vocab)}')
print(f'中文词汇表大小：{len(zh_vocab)}')


# 将句子转换为索引序列
def process_sentence(sentence, tokenizer, vocab):
    """
    将句子转换为索引序列，并添加 <bos> 和 <eos>
    :param sentence: 输入句子
    :param tokenizer: 分词器函数
    :param vocab: 对应的词汇表
    :return: 索引序列
    """
    tokens = tokenizer(sentence)
    tokens = ['<bos>'] + tokens + ['<eos>']
    indices = [vocab[token] for token in tokens]
    return indices


# 将所有句子转换为索引序列
en_sequences = [process_sentence(sentence, tokenizer_en, en_vocab) for sentence in english_sentences]
zh_sequences = [process_sentence(sentence, tokenizer_zh, zh_vocab) for sentence in chinese_sentences]
# 查看示例句子的索引序列
print("示例英文句子索引序列：", en_sequences[0])
print("示例中文句子索引序列：", zh_sequences[0])


# 创建数据集和数据加载器
class TranslationDataset(Dataset):
    def __init__(self, src_sequences, trg_sequences):
        self.src_sequences = src_sequences
        self.trg_sequences = trg_sequences

    def __len__(self):
        return len(self.src_sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.src_sequences[idx]), torch.tensor(self.trg_sequences[idx])


# 填充对齐操作
def collate_fn(batch):
    """
    自定义的 collate_fn，用于将批次中的样本进行填充对齐
    """
    src_batch, trg_batch = [], []
    for src_sample, trg_sample in batch:
        src_batch.append(src_sample)
        trg_batch.append(trg_sample)
    src_batch = pad_sequence(src_batch, padding_value=en_vocab['<pad>'], batch_first=True)
    trg_batch = pad_sequence(trg_batch, padding_value=zh_vocab['<pad>'], batch_first=True)
    return src_batch, trg_batch

# 创建数据集对象
dataset = TranslationDataset(en_sequences, zh_sequences)
# 划分训练集和验证集
train_data, val_data = train_test_split(dataset, test_size=0.1)
# 创建数据加载器
batch_size = 32
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


# Step 3: Transformer模型
class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm_1 = LayerNormalization()
        self.multihead_attention = MultiheadAttention()
        self.drop_out_1 = nn.Dropout(0.1)

        self.layer_norm_2 = LayerNormalization()
        self.feed_forward = FeedForwardLayer()
        self.drop_out_2 = nn.Dropout(0.1)

    def forward(self, x, e_mask):
        x_1 = self.layer_norm_1(x)  # (B, L, d_model)
        x = x + self.drop_out_1(
            self.multihead_attention(x_1, x_1, x_1, mask=e_mask)
        )  # (B, L, d_model)
        x_2 = self.layer_norm_2(x)  # (B, L, d_model)
        x = x + self.drop_out_2(self.feed_forward(x_2))  # (B, L, d_model)
        return x  # (B, L, d_model)


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm_1 = LayerNormalization()
        self.masked_multihead_attention = MultiheadAttention()
        self.drop_out_1 = nn.Dropout(0.1)

        self.layer_norm_2 = LayerNormalization()
        self.multihead_attention = MultiheadAttention()
        self.drop_out_2 = nn.Dropout(0.1)

        self.layer_norm_3 = LayerNormalization()
        self.feed_forward = FeedForwardLayer()
        self.drop_out_3 = nn.Dropout(0.1)

    def forward(self, x, e_output, e_mask,  d_mask):
        x_1 = self.layer_norm_1(x)  # (B, L, d_model)
        x = x + self.drop_out_1(
            self.masked_multihead_attention(x_1, x_1, x_1, mask=d_mask)
        ) # (B, L, d_model)
        x_2 = self.layer_norm_2(x)  # (B, L, d_model)
        x = x + self.drop_out_2(
            self.multihead_attention(x_2, e_output, e_output, mask=e_mask)
        ) # (B, L, d_model)
        x_3 = self.layer_norm_3(x)  # (B, L, d_model)
        x = x + self.drop_out_3(self.feed_forward(x_3))  # (B, L, d_model)

        return x  # (B, L, d_model)


class MultiheadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.inf = 1e9

        # W^Q, W^K, W^V in the paper
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(0.1)
        self.attn_softmax = nn.Softmax(dim=-1)

        # Final output linear transformation
        self.w_0 = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        input_shape = q.shape

        # Linear calculation +  split into num_heads
        q = self.w_q(q).view(input_shape[0], -1, num_heads, 64)  # (B, L, num_heads, d_k)
        k = self.w_k(k).view(input_shape[0], -1, num_heads, 64)  # (B, L, num_heads, d_k)
        v = self.w_v(v).view(input_shape[0], -1, num_heads, 64)  # (B, L, num_heads, d_k)

        # For convenience, convert all tensors in size (B, num_heads, L, d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Conduct self-attention
        attn_values = self.self_attention(q, k, v, mask=mask)  # (B, num_heads, L, d_k)
        concat_output = attn_values.transpose(1, 2)\
            .contiguous().view(input_shape[0], -1, d_model)  # (B, L, d_model)

        return self.w_0(concat_output)

    def self_attention(self, q, k, v, mask=None):
        # Calculate attention scores with scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # (B, num_heads, L, L)
        attn_scores = attn_scores / math.sqrt(64)

        # If there is a mask, make masked spots -INF
        if mask is not None:
            mask = mask.unsqueeze(1)  # (B, 1, L) => (B, 1, 1, L) or (B, L, L) => (B, 1, L, L)
            attn_scores = attn_scores.masked_fill_(mask == 0, -1 * self.inf)

        # Softmax and multiplying K to calculate attention value
        attn_distribs = self.attn_softmax(attn_scores)

        attn_distribs = self.dropout(attn_distribs)
        attn_values = torch.matmul(attn_distribs, v)  # (B, num_heads, L, d_k)

        return attn_values


class FeedForwardLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff, bias=True)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.relu(self.linear_1(x))  # (B, L, d_ff)
        x = self.dropout(x)
        x = self.linear_2(x)  # (B, L, d_model)
        return x


class LayerNormalization(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.layer = nn.LayerNorm([d_model], elementwise_affine=True, eps=self.eps)

    def forward(self, x):
        x = self.layer(x)
        return x


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model

        # Make initial positional encoding matrix with 0
        pe_matrix = torch.zeros(max_len, d_model)  # (L, d_model)

        # Calculating position encoding values
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pe_matrix[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe_matrix[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))

        self.register_buffer('pe_matrix', pe_matrix)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)  # (B, L, d_model)
        # Add positional encoding to the input
        x = x + self.pe_matrix[:x.size(1)].unsqueeze(0)  # (B, L, d_model)
        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size

        self.src_embedding = nn.Embedding(self.src_vocab_size, d_model)
        self.trg_embedding = nn.Embedding(self.trg_vocab_size, d_model)
        self.positional_encoder = PositionalEncoder(d_model=512)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.output_linear = nn.Linear(d_model, self.trg_vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, src_input, trg_input, e_mask=None, d_mask=None):
        src_input = self.src_embedding(src_input)  # (B, L) => (B, L, d_model)
        trg_input = self.trg_embedding(trg_input)  # (B, L) => (B, L, d_model)
        src_input = self.positional_encoder(src_input)  # (B, L, d_model) => (B, L, d_model)
        trg_input = self.positional_encoder(trg_input)  # (B, L, d_model) => (B, L, d_model)

        e_output = self.encoder(src_input, e_mask)  # (B, L, d_model)
        d_output = self.decoder(trg_input, e_output, e_mask, d_mask)  # (B, L, d_model)

        output = self.softmax(self.output_linear(d_output))  # (B, L, d_model) => # (B, L, trg_vocab_size)
        return output


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer() for i in range(num_layers)])
        self.layer_norm = LayerNormalization()

    def forward(self, x, e_mask):
        for i in range(num_layers):
            x = self.layers[i](x, e_mask)

        return self.layer_norm(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer() for i in range(num_layers)])
        self.layer_norm = LayerNormalization()

    def forward(self, x, e_output, e_mask, d_mask):
        for i in range(num_layers):
            x = self.layers[i](x, e_output, e_mask, d_mask)

        return self.layer_norm(x)

# 初始化模型参数
input_dim = len(en_vocab)
output_dim = len(zh_vocab)
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 3
dropout = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 实例化编码器、解码器和 Transformer 模型
encoder = Encoder()
decoder = Decoder()
model = Transformer(input_dim, output_dim).to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss(ignore_index=zh_vocab['<pad>'])
optimizer = optim.Adam(params=model.parameters(), lr=0.0001)


# Step 4: 训练
# 训练函数
def train(model, dataloader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for src, trg in dataloader:
        src = src.to(device)
        trg = trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg[:, :-1])  # 输入不包括最后一个词
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)  # 目标不包括第一个词
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

# 验证函数
def evaluate(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, trg in dataloader:
            src = src.to(device)
            trg = trg.to(device)
            output = model(src, trg[:, :-1])
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

# 开始训练
n_epochs = 10
for epoch in range(n_epochs):
    train_loss = train(model, train_dataloader, optimizer, criterion)
    val_loss = evaluate(model, val_dataloader, criterion)
    print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}')


# Step 5: 测试与推理
# 测试句子
test_sentence = "How are you?"

# 处理测试句子为索引
test_sentence_indices = process_sentence(test_sentence, tokenizer_en, en_vocab)
test_input = torch.tensor([test_sentence_indices]).to(device)

# 设置解码器的初始输入为 <bos>
start_token = torch.tensor([[zh_vocab['<bos>']]]).to(device)

# 使用模型进行推理
model.eval()
with torch.no_grad():
    src_mask = (test_input != en_vocab['<pad>']).unsqueeze(1).unsqueeze(2).to(device)  # 忽略填充的元素
    trg_input = start_token  # 目标语言的初始输入是 <bos> token
    predicted_indices = []

    # 逐词生成翻译
    for _ in range(50):  # 假设最大生成长度为 50
        output = model(test_input, trg_input)  # (B, L, vocab_size)
        output = output[:, -1, :]  # 取出最后一个位置的输出
        pred_token = output.argmax(dim=-1).unsqueeze(1)  # 选择概率最大的token
        predicted_indices.append(pred_token.item())

        # 将预测的token添加到目标输入中
        trg_input = torch.cat([trg_input, pred_token], dim=1)

        # 如果预测到 <eos>，则停止生成
        if pred_token.item() == zh_vocab['<eos>']:
            break

# 将预测的索引转换为中文句子
predicted_sentence = ''.join([zh_vocab.get_itos()[idx] for idx in predicted_indices])
print(f'Original English sentence: "{test_sentence}"')
print(f'Translated Chinese sentence: "{predicted_sentence}"')
