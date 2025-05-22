import torch
import torch.nn as nn
import torch.nn.functional as F


class CharacterEncoder(nn.Module):
    def __init__(self, char_vocab_size, char_embed_dim, conv_channels, kernel_sizes):
        super(CharacterEncoder, self).__init__()
        self.char_embed = nn.Embedding(char_vocab_size, char_embed_dim)
        self.convs = nn.ModuleList([nn.Conv1d(char_embed_dim, conv_channels, kernel_size) for kernel_size in kernel_sizes])

    def forward(self, char_seq):
        embedded = self.char_embed(char_seq)  # (batch_size, word_len, char_embed_dim)
        embedded = embedded.permute(0, 2, 1)  # (batch_size, char_embed_dim, word_len)

        conv_outputs = []
        for conv in self.convs:
            conv_output = F.relu(conv(embedded))  # (batch_size, conv_channels, seq_len - kernel_size + 1)
            pooled = F.max_pool1d(conv_output, conv_output.size(2)).squeeze(2)  # (batch_size, conv_channels)
            conv_outputs.append(pooled)

        output = torch.cat(conv_outputs, dim=1)  # (batch_size, num_convs * conv_channels)
        return output


class ELMo(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout_prob, char_vocab_size, char_embed_dim, conv_channels, kernel_sizes):
        super(ELMo, self).__init__()
        self.char_encoder = CharacterEncoder(char_vocab_size, char_embed_dim, conv_channels, kernel_sizes)
        self.word_embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim + conv_channels * len(kernel_sizes), hidden_size=hidden_dim, num_layers=num_layers,
                            bidirectional=True, dropout=dropout_prob, batch_first=True)

    def forward(self, word_seq, char_seq):
        # word_seq: (batch_size, seq_len)
        # char_seq: (batch_size, seq_len, word_len)
        batch_size, seq_len, word_len = char_seq.shape

        # Reshape for character encoding
        char_seq = char_seq.view(batch_size * seq_len, word_len)  # (batch_size * seq_len, word_len)
        char_embeddings = self.char_encoder(char_seq)  # (batch_size * seq_len, num_convs * conv_channels)
        char_embeddings = char_embeddings.view(batch_size, seq_len, -1)  # (batch_size, seq_len, num_convs * conv_channels)

        word_embeddings = self.word_embed(word_seq)  # (batch_size, seq_len, embedding_dim)

        # Concatenate word and character embeddings
        combined_embeddings = torch.cat((word_embeddings, char_embeddings), dim=2)  # (batch_size, seq_len, embedding_dim + num_convs * conv_channels)

        lstm_out, _ = self.lstm(combined_embeddings)  # (batch_size, seq_len, 2 * hidden_dim)

        return lstm_out


# 参数设置
vocab_size = 10000
embedding_dim = 128
hidden_dim = 256
num_layers = 2
dropout_prob = 0.1
char_vocab_size = 100
char_embed_dim = 16
conv_channels = 128
kernel_sizes = [2, 3, 4, 5]


elmo = ELMo(vocab_size, embedding_dim, hidden_dim, num_layers, dropout_prob, char_vocab_size, char_embed_dim, conv_channels, kernel_sizes)

word_seq = torch.randint(0, vocab_size, (2, 10))
char_seq = torch.randint(0, char_vocab_size, (2, 10, 15))


output = elmo(word_seq, char_seq)
print(output.shape)  # (batch_size, sequence_length, 2 * hidden_dim) = (2, 10, 512)
