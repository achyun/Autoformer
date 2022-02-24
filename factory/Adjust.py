import torch.nn as nn
import torch.nn.functional as F
from .Norm import ConvNorm, LinearNorm


class Adjust(nn.Module):
    def __init__(self, dim_emb, dim_cell=768):
        super(Adjust, self).__init__()

        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(
                    80 + dim_emb if i == 0 else 512,
                    512,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    dilation=1,
                    w_init_gain="relu",
                ),
                nn.BatchNorm1d(512),
            )
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(512, hidden_size=dim_cell, num_layers=3, batch_first=True)
        self.embedding = LinearNorm(dim_cell, 256)

    def forward(self, x):
        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)
        x = self.lstm(x)[0][:, -1, :]
        embeds = self.embedding(x)
        norm = embeds.norm(p=2, dim=-1, keepdim=True)
        return embeds.div(norm)
