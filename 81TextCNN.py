import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, max_sequence_length, max_token_num, embedding_dim, output_dim, embedding_matrix=None):
        super(TextCNN, self).__init__()
        self.max_sequence_length = max_sequence_length
        self.max_token_num = max_token_num
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim

        if embedding_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        else:
            self.embedding = nn.Embedding(max_token_num, embedding_dim)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=2, kernel_size=ks) for ks in [2, 3, 4]
        ])

        self.fc = nn.Linear(2 * 3, output_dim)

    def forward(self, x):
        # (batch_size, max_sequence_length)
        x = self.embedding(x)  # (batch_size, max_sequence_length, embedding_dim)
        x = x.permute(0, 2, 1)  # (batch_size, embedding_dim, max_sequence_length)

        pool_output = []
        for conv in self.convs:
            c = conv(x)  # (batch_size, 2, output_length)
            p = F.max_pool1d(c, kernel_size=c.size(2))  # (batch_size, 2, 1)
            pool_output.append(p)

        pool_output = torch.cat(pool_output, dim=1)  # (batch_size, 2*3, 1)
        pool_output = pool_output.squeeze(2)  # (batch_size, 2*3)

        x = self.fc(pool_output)  # (batch_size, output_dim)
        x = F.softmax(x, dim=1)

        return x


if __name__ == "__main__":
    max_sequence_length = 60
    max_token_num = 10000
    embedding_dim = 300
    output_dim = 2
    embedding_matrix = None

    model = TextCNN(max_sequence_length, max_token_num, embedding_dim, output_dim, embedding_matrix)
    x = torch.randint(0, max_token_num, (32, max_sequence_length))
    output = model(x)

    print(output.shape)

    # 打印模型结构
    # from torchsummary import summary
    #
    # summary(model, input_size=(max_sequence_length,))
