import torch
import torch.nn as nn
import torch.nn.functional as F


class fastText(nn.Module):
    def __init__(self, vocab_size, twoGrams_size, threeGrams_size, embed_size, hidden_size, output_size, embedding_pretrained=None):
        super(fastText, self).__init__()

        # Embedding layer
        if embedding_pretrained is None:
            self.embedding_word = nn.Embedding(vocab_size, embed_size)
        # 使用预训练词向量
        else:
            self.embedding_word = nn.Embedding.from_pretrained(embedding_pretrained, freeze=False)
        # self.embedding_word.weight.requires_grad = True
        self.embedding_2gram = nn.Embedding(twoGrams_size, embed_size)
        self.embedding_3gram = nn.Embedding(threeGrams_size, embed_size)
        self.dropout = nn.Dropout(p=0.5)

        # Hidden layer
        self.hidden = nn.Linear(embed_size, hidden_size)
        # Output layer
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Args: Tensor
            x[0]: word
            x[1]: 2grams
            x[2]: 3grams
        """
        # x: (word, 2_gram, 3_gram), word, 2_gram和3_gram形状都是(batch_size, *)
        e_word = self.embedding_word(x[0])  # e_word: (batch_size, seq_len_word, embed_size)
        e_2gram = self.embedding_2gram(x[1])  # e_2gram: (batch_size, seq_len_2gram, embed_size)
        e_3gram = self.embedding_3gram(x[2])  # e_3gram: (batch_size, seq_len_3gram, embed_size)
        e_cat = torch.cat((e_word, e_2gram, e_3gram), dim=1)
        e_avg = e_cat.mean(dim=1)
        h = self.hidden(self.dropout(e_avg))  # input: (batch_size, embed_size), h:(batch_size, hidden_size)
        o = F.softmax(self.output(h), dim=1)  # o: (batch_size, output_size)
        return o, {
            "embedding_word": e_word,
            "embedding_2gram": e_2gram,
            "embedding_3gram": e_3gram,
            "e_cat": e_cat,
            "e_avg": e_avg,
            "hidden": h
        }


vocab_size = 10
twoGrams_size = 20
threeGrams_size = 30
embed_size = 128
hidden_size = 256
output_size = 16
ft = fastText(vocab_size, twoGrams_size, threeGrams_size, embed_size, hidden_size, output_size)
print(ft)

x_0 = torch.LongTensor([[1, 2, 3, 3, 5]])  # batch_size=1, seq_len=5
x_1 = torch.LongTensor([[1, 2, 3, 4]])  # batch_size=1, seq_len=4
x_2 = torch.LongTensor([[1, 2, 3]])  # batch_size=1, seq_len=3
x = (x_0, x_1, x_2)
output, tmp = ft(x)
print("embedding_word:", tmp["embedding_word"].size())
print("embedding_2gram:", tmp["embedding_2gram"].size())
print("embedding_3gram:", tmp["embedding_3gram"].size())
print("e_cat:", tmp["e_cat"].size())
print("e_avg:", tmp["e_avg"].size())
print("hidden:", tmp["hidden"].size())
print("output", output.size())
