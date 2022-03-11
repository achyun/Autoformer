import torch
import torch.nn as nn
import random
from .MLPMixer import *
from .Norm import ConvNorm


class MetaBlock(nn.Module):
    def __init__(
        self,
        num_classes,
        dim_emb=256,
        emb=768,
        crop_len=128,
        patch_size=32,
        mlp_depth=1,
        kernel_size=3,
        channels=1,
        sample_count=4,
    ):

        super().__init__()
        self.token_mixer = nn.LSTM(
            input_size=80, hidden_size=emb, num_layers=1, batch_first=True,
        )
        self.conv_1 = ConvNorm(crop_len, emb, kernel_size=kernel_size)
        self.mlp = MLPMixer(
            image_size=emb,
            channels=channels,
            patch_size=patch_size,
            dim=emb,
            depth=mlp_depth,
        )
        self.dv = nn.Linear(emb * 2, dim_emb)
        self.output = nn.Linear(dim_emb, num_classes)
        self.crop_len = crop_len
        self.sample_count = sample_count

    def forward(self, x):
        x, _ = self.token_mixer(x)
        embedding_1 = x[:, -1, :]
        embedding_2 = []
        for _ in range(self.sample_count):
            left = random.randint(0, x.shape[1] - self.crop_len)
            x_after_conv_1 = self.conv_1(x[:, left : left + self.crop_len, :])
            x_mlp = self.mlp(x_after_conv_1.unsqueeze(1))
            embedding_2.append(x_mlp[:, -1, :])

        embedding_2 = torch.stack(embedding_2).sum(dim=0).div(self.sample_count)
        embedding = self.dv(torch.cat((embedding_1, embedding_2), dim=1))
        norm = embedding.norm(p=2, dim=-1, keepdim=True)
        d_vec = embedding.div(norm)
        predictions = self.output(embedding)
        return predictions, d_vec


class MetaDV(nn.Module):
    """
    For Gan Training Use
    """

    def __init__(self, num_classes=256):
        super(MetaDV, self).__init__()
        self.metablock = MetaBlock(num_classes)

    def forward(self, x):
        return self.metablock(x)
