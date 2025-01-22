import json
import os
import sys
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# Transformer Decoder
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, q, k, v, attention_mask):
        ##
        # q: [batch_size, n_heads, len_q, d_k]
        # k: [batch_size, n_heads, len_k, d_k]
        # v: [batch_size, n_heads, len_v, d_v]
        # attn_mask: [batch_size, n_heads, seq_len, seq_len]
        ##
        # 计算每个Q与K的分数，计算出来的大小是 [batch_size, n_heads, len_q, len_q]
        scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.d_k)
        # 把被mask的地方置为无限小，softmax之后基本就是0，也就对q不起作用
        scores.masked_fill_(attention_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        # 注意力后的大小 [batch_size, n_heads, len_q, d_v]
        context = torch.matmul(attn, v)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.w_q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.w_k = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.w_v = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, attention_mask):
        ##
        # q: [batch_size, seq_len, d_model]
        # k: [batch_size, seq_len, d_model]
        # v: [batch_size, seq_len, d_model]
        # attn_mask: [batch_size, seq_len, seq_len]
        ##
        # 记录原始值, 后续计算残差
        residual, batch_size = q, q.size(0)
        # 先映射 q、k、v, 然后后分头
        # q: [batch_size, n_heads, len_q, d_k]
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # k: [batch_size, n_heads, len_k, d_k]
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # v: [batch_size, n_heads, len_v(=len_k), d_v]
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        attention_mask = attention_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        # 点积注意力分数计算，  [batch_size, n_heads, len_q, d_v]
        context, attn = ScaledDotProductAttention(self.d_k)(q, k, v, attention_mask)
        # context: [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)
        # 还原为原始大小
        output = self.fc(context)
        # LN + 残差计算
        return self.layernorm(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        ##
        # inputs: [batch_size, seq_len, d_model]
        ##
        residual = inputs
        output = self.fc(inputs)
        # # LN + 残差计算, [batch_size, seq_len, d_model]
        return self.layernorm(output + residual)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, d_k, d_v):
        super(DecoderLayer, self).__init__()
        # 多头注意力层
        self.attention = MultiHeadAttention(d_model, n_heads, d_k, d_v)
        # 前馈神经网络层
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, inputs, attention_mask):
        ##
        # inputs: [batch_size, seq_len, d_model]
        # attention_mask: [batch_size, seq_len, seq_len]
        ##
        # outputs: [batch_size, seq_len, d_model]
        # self_attn: [batch_size, n_heads, seq_len, seq_len]
        outputs, self_attn = self.attention(inputs, inputs, inputs, attention_mask)
        # [batch_size, seq_len, d_model]
        outputs = self.pos_ffn(outputs)
        return outputs, self_attn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_pos, device):
        super(PositionalEncoding, self).__init__()
        self.device = device
        self.pos_embedding = nn.Embedding(max_pos, d_model)

    def forward(self, inputs):
        seq_len = inputs.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=self.device)
        # [seq_len] -> [batch_size, seq_len]
        pos = pos.unsqueeze(0).expand_as(inputs)
        return self.pos_embedding(pos)

def get_attn_subsequence_mask(seq, device):
    # 注意力分数的大小是 [batch_size, n_heads, len_seq, len_seq]
    # 所以这里要生成 [batch_size, len_seq, len_seq] 大小
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # 生成一个上三角矩阵
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    subsequence_mask = subsequence_mask.to(device)
    return subsequence_mask

def get_attn_pad_mask(attention_mask):
    batch_size, len_seq = attention_mask.size()
    attention_mask = attention_mask.data.eq(0).unsqueeze(1)
    # 注意力分数的大小是 [batch_size, n_heads, len_q, len_q]
    # 所以这里要转换成 [batch_size, len_seq, len_seq] 大小
    return attention_mask.expand(batch_size, len_seq, len_seq)

class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, d_k, d_v, vocab_size, max_pos, n_layers, device):
        super(Decoder, self).__init__()
        self.device = device
        # 将Token转为向量
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_pos, device)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, d_k, d_v) for _ in range(n_layers)])

    def forward(self, inputs, attention_mask):
        ##
        # inputs: [batch_size, seq_len]
        ##
        # [batch_size, seq_len, d_model]
        outputs = self.embedding(inputs) + self.pos_encoding(inputs)
        # 上三角掩码，防止看到未来的信息， [batch_size, seq_len, seq_len]
        subsequence_mask = get_attn_subsequence_mask(inputs, self.device)
        if attention_mask is not None:
            # pad掩码 [batch_size, seq_len, seq_len]
            attention_mask = get_attn_pad_mask(attention_mask)
            # [batch_size, seq_len, seq_len]
            attention_mask = torch.gt((attention_mask + subsequence_mask), 0)
        else:
            attention_mask = subsequence_mask.bool()
        # 计算每一层的结果
        self_attns = []
        for layer in self.layers:
            # outputs: [batch_size, seq_len, d_model],
            # self_attn: [batch_size, n_heads, seq_len, seq_len],
            outputs, self_attn = layer(outputs, attention_mask)
            self_attns.append(self_attn)
        return outputs, self_attns


# GPT
class GPTModel(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, d_k, d_v, vocab_size, max_pos, n_layers, device):
        super(GPTModel, self).__init__()
        # 解码器
        self.decoder = Decoder(d_model, n_heads, d_ff, d_k, d_v, vocab_size, max_pos, n_layers, device)
        # 映射为词表大小
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, inputs, attention_mask=None):
        ##
        # inputs: [batch_size, seq_len]
        ##
        # outputs: [batch_size, seq_len, d_model]
        # self_attns: [n_layers, batch_size, n_heads, seq_len, seq_len]
        outputs, self_attns = self.decoder(inputs, attention_mask)
        # [batch_size, seq_len, vocab_size]
        logits = self.projection(outputs)
        return logits.view(-1, logits.size(-1)), self_attns

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 模型参数
    model_param = {
        "d_model": 768,  # 嵌入层大小
        "d_ff": 2048,  # 前馈神经网络大小
        "d_k": 64,  # K 的大小
        "d_v": 64,  # V 的大小
        "n_layers": 6,  # 解码层的数量
        "n_heads": 8,  # 多头注意力的头数
        "max_pos": 1800,  # 位置编码的长度
        "device": device,  # 设备
        "vocab_size": 4825  # 词表大小
    }
    model = GPTModel(**model_param)
    total_params = sum(p.numel() for p in model.parameters())
    print(model)
    print("total_params: ", total_params)


# 数据集
def build_vocab(file_path):
    # 读取所有文本
    texts = []
    with open(file_path, 'r', encoding='utf-8') as r:
        for line in r:
            if not line:
                continue
            line = json.loads(line)
            question = line["question"]
            answer = line["answer"]
            texts.append(question)
            texts.append(answer)
    # 拆分 Token
    words = set()
    for t in texts:
        if not t:
            continue
        for word in t.strip():
            words.add(word)
    words = list(words)
    words.sort()
    # 特殊Token
    # pad 占位、unk 未知、sep 结束
    word2id = {"<pad>": 0, "<unk>": 1, "<sep>": 2}
    # 构建词表
    word2id.update({word: i + len(word2id) for i, word in enumerate(words)})
    id2word = list(word2id.keys())
    vocab = {"word2id": word2id, "id2word": id2word}
    vocab = json.dumps(vocab, ensure_ascii=False)
    with open('./datasets/GPT/vocab.json', 'w', encoding='utf-8') as w:
        w.write(vocab)
    print(f"finish. words: {len(id2word)}")

build_vocab("./datasets/GPT/train.jsonl")

# Tokenizer
class Tokenizer:
    def __init__(self, vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as r:
            vocab = r.read()
            if not vocab:
                raise Exception("词表读取为空！")
        vocab = json.loads(vocab)
        self.word2id = vocab["word2id"]
        self.id2word = vocab["id2word"]
        self.pad_token = self.word2id["<pad>"]
        self.unk_token = self.word2id["<unk>"]
        self.sep_token = self.word2id["<sep>"]

    def encode(self, text, text1=None, max_length=128, pad_to_max_length=False):
        tokens = [self.word2id[word] if word in self.word2id else self.unk_token for word in text]
        tokens.append(self.sep_token)
        if text1:
            tokens.extend([self.word2id[word] if word in self.word2id else self.unk_token for word in text1])
            tokens.append(self.sep_token)
        att_mask = [1] * len(tokens)
        if pad_to_max_length:
            if len(tokens) > max_length:
                tokens = tokens[0:max_length]
                att_mask = att_mask[0:max_length]
            elif len(tokens) < max_length:
                tokens.extend([self.pad_token] * (max_length - len(tokens)))
                att_mask.extend([0] * (max_length - len(att_mask)))
        return tokens, att_mask

    def decode(self, token):
        if type(token) is tuple or type(token) is list:
            return [self.id2word[n] for n in token]
        else:
            return self.id2word[token]

    def get_vocab_size(self):
        return len(self.id2word)

tokenizer = Tokenizer(vocab_path="./datasets/GPT/vocab.json")
encode, att_mask = tokenizer.encode("你好", "你好", pad_to_max_length=True)
decode = tokenizer.decode(encode)
print("token lens: ", len(encode))
print("encode: ", encode)
print("att_mask: ", att_mask)
print("decode: ", decode)
print("vocab_size", tokenizer.get_vocab_size())

def split_dataset(file_path, output_path):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    datas = []
    with open(file_path, "r", encoding='utf-8') as f:
        for line in f:
            if not line or line == "":
                continue
            datas.append(line)
    train = datas[0:10000]
    val = datas[10000:11000]
    with open(os.path.join(output_path, "train.json"), "w", encoding="utf-8") as w:
        for line in train:
            w.write(line)
            w.flush()

    with open(os.path.join(output_path, "val.json"), "w", encoding="utf-8") as w:
        for line in val:
            w.write(line)
            w.flush()
    print("train count: ", len(train))
    print("val count: ", len(val))

file_path = "./datasets/GPT/train.jsonl"
split_dataset(file_path=file_path, output_path="./datasets/GPT/data")


# 训练
class QADataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        if data_path:
            with open(data_path, "r", encoding='utf-8') as f:
                for line in f:
                    if not line or line == "":
                        continue
                    json_line = json.loads(line)
                    question = json_line["question"]
                    answer = json_line["answer"]
                    self.data.append({
                        "question": question,
                        "answer": answer
                    })
        print("data load ， size：", len(self.data))

    def preprocess(self, question, answer):
        encode, att_mask = self.tokenizer.encode(question, answer, max_length=self.max_length, pad_to_max_length=True)
        input_ids = encode[:-1]
        att_mask = att_mask[:-1]
        labels = encode[1:]
        return input_ids, att_mask, labels

    def __getitem__(self, index):
        item_data = self.data[index]
        input_ids, att_mask, labels = self.preprocess(**item_data)
        return {
            "input_ids": torch.LongTensor(np.array(input_ids)),
            "attention_mask": torch.LongTensor(np.array(att_mask)),
            "labels": torch.LongTensor(np.array(labels))
        }

    def __len__(self):
        return len(self.data)


def train_model(model, train_loader, val_loader, optimizer, criterion,
                device, num_epochs, model_output_dir, writer):
    batch_step = 0
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        time1 = time.time()
        model.train()
        for index, data in enumerate(tqdm(train_loader, file=sys.stdout, desc="Train Epoch: " + str(epoch))):
            input_ids = data['input_ids'].to(device, dtype=torch.long)
            attention_mask = data['attention_mask'].to(device, dtype=torch.long)
            labels = data['labels'].to(device, dtype=torch.long)
            optimizer.zero_grad()
            outputs, dec_self_attns = model(input_ids, attention_mask)
            loss = criterion(outputs, labels.view(-1))
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            writer.add_scalar('Loss/train', loss, batch_step)
            batch_step += 1
            # 100轮打印一次 loss
            if index % 100 == 0 or index == len(train_loader) - 1:
                time2 = time.time()
                tqdm.write(
                    f"{index}, epoch: {epoch} -loss: {str(loss)} ; lr: {optimizer.param_groups[0]['lr']} ;each step's time spent: {(str(float(time2 - time1) / float(index + 0.0001)))}")
        # 验证
        model.eval()
        val_loss = validate_model(model, criterion, device, val_loader)
        writer.add_scalar('Loss/val', val_loss, epoch)
        print(f"val loss: {val_loss} , epoch: {epoch}")
        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(model_output_dir, "best.pt")
            print("Save Best Model To ", best_model_path, ", epoch: ", epoch)
            torch.save(model.state_dict(), best_model_path)
        # 保存当前模型
        last_model_path = os.path.join(model_output_dir, "last.pt")
        print("Save Last Model To ", last_model_path, ", epoch: ", epoch)
        torch.save(model.state_dict(), last_model_path)


def validate_model(model, criterion, device, val_loader):
    running_loss = 0.0
    with torch.no_grad():
        for _, data in enumerate(tqdm(val_loader, file=sys.stdout, desc="Validation Data")):
            input_ids = data['input_ids'].to(device, dtype=torch.long)
            attention_mask = data['attention_mask'].to(device, dtype=torch.long)
            labels = data['labels'].to(device, dtype=torch.long)
            outputs, dec_self_attns = model(input_ids, attention_mask)
            loss = criterion(outputs, labels.view(-1))
            running_loss += loss.item()
    return running_loss / len(val_loader)


def train_main():
    train_json_path = "./datasets/GPT/data/train.json"  # 训练集
    val_json_path = "./datasets/GPT/data/val.json"  # 验证集
    vocab_path = "./datasets/GPT/vocab.json"  # 词表位置
    max_length = 120  # 最大长度
    epochs = 15  # 迭代周期
    batch_size = 128  # 训练一个批次的大小
    lr = 1e-4  # 学习率
    model_output_dir = "model_output"  # 模型保存目录
    logs_dir = "logs"  # 日志记录目标
    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 加载分词器
    tokenizer = Tokenizer(vocab_path)
    # 模型参数
    model_param = {
        "d_model": 768,  # 嵌入层大小
        "d_ff": 2048,  # 前馈神经网络大小
        "d_k": 64,  # K 的大小
        "d_v": 64,  # V 的大小
        "n_layers": 6,  # 解码层的数量
        "n_heads": 8,  # 多头注意力的头数
        "max_pos": 1800,  # 位置编码的长度
        "device": device,  # 设备
        "vocab_size": tokenizer.get_vocab_size(),  # 词表大小
    }
    model = GPTModel(**model_param)
    print("Start Load Train Data...")
    train_params = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": 4,
    }
    training_set = QADataset(train_json_path, tokenizer, max_length)
    training_loader = DataLoader(training_set, **train_params)
    print("Start Load Validation Data...")
    val_params = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": 4,
    }
    val_set = QADataset(val_json_path, tokenizer, max_length)
    val_loader = DataLoader(val_set, **val_params)
    # 日志记录
    writer = SummaryWriter(logs_dir)
    # 优化器
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr)
    # 损失函数
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)
    model = model.to(device)
    # 开始训练
    print("Start Training...")
    train_model(
        model=model,
        train_loader=training_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=epochs,
        model_output_dir=model_output_dir,
        writer=writer
    )
    writer.close()


# 预测
def generate(model, tokenizer, text, max_length, device):
    input, att_mask = tokenizer.encode(text)
    input = torch.tensor(input, dtype=torch.long, device=device).unsqueeze(0)
    stop = False
    input_len = len(input[0])
    while not stop:
        if len(input[0]) - input_len > max_length:
            next_symbol = tokenizer.sep_token
            input = torch.cat(
                [input.detach(), torch.tensor([[next_symbol]], dtype=input.dtype, device=device)], -1)
            break
        projected, self_attns = model(input)
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[-1]
        next_symbol = next_word
        if next_symbol == tokenizer.sep_token:
            stop = True
        input = torch.cat(
            [input.detach(), torch.tensor([[next_symbol]], dtype=input.dtype, device=device)], -1)
    decode = tokenizer.decode(input[0].tolist())
    decode = decode[len(text):]
    return "".join(decode)

def predict_main():
    vocab_path = "./datasets/GPT/vocab.json"  # 词表位置
    max_length = 128  # 最大长度
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 加载分词器
    tokenizer = Tokenizer(vocab_path)
    # 模型参数
    model_param = {
        "d_model": 768,  # 嵌入层大小
        "d_ff": 2048,  # 前馈神经网络大小
        "d_k": 64,  # K 的大小
        "d_v": 64,  # V 的大小
        "n_layers": 6,  # 解码层的数量
        "n_heads": 8,  # 多头注意力的头数
        "max_pos": 1800,  # 位置编码的长度
        "device": device,  # 设备
        "vocab_size": tokenizer.get_vocab_size(),  # 词表大小
    }
    model = GPTModel(**model_param)
    model.to(device)

    while True:
        text = input("请输入：")
        if not text:
            continue
        if text == "q":
            break
        res = generate(model, tokenizer, text, max_length, device)
        print("AI: ", res)


if __name__ == '__main__':
    train_main()
    predict_main()
