import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import Config


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.patch_embedding = PatchEmbedding()
        self.class_token = nn.Parameter(torch.randn(Config.batch_size, 1, Config.embedding_d), requires_grad=True)
        self.position_embedding = nn.Parameter(torch.randn(1, Config.num_patches + 1, Config.embedding_d),
                                               requires_grad=True)
        self.dropout = nn.Dropout(Config.dropout_rate)
        self.patch_embedding = PatchEmbedding()
        self.transformer_encoder = nn.ModuleList([Encoder() for _ in range(12)])
        self.classifier = nn.Sequential(
            nn.LayerNorm(Config.embedding_d),
            nn.Linear(Config.embedding_d, Config.num_classes)
        )

    def forward(self, x):
        # print(x.shape)  # batch * 1 * 29 * 29

        x = F.pad(x, (1, 2, 1, 2))
        # print(x.shape)  # batch * 1 * 32 * 32

        x = self.patch_embedding(x)
        # print(x.shape)  # b * 16 * 768

        class_token = self.class_token.expand(Config.batch_size, -1, -1)
        # print(class_token.shape)  # b * 1 * 768

        x = torch.cat((class_token, x), dim=1)
        # print(x.shape)  # b * 17 * 768

        pe = self.position_embedding
        # print(pe.shape)  # 1 * 17 * 768

        x = pe + x
        # print(x.shape)  # 32 * 17 * 768

        x = self.dropout(x)

        for encoder_layer in self.transformer_encoder:
            x = encoder_layer(x)
        # print(x.shape)  # batch * 17 * 768

        x = self.classifier(x[:, 0])
        # print(x.shape)  # batch * 2

        return x


class PatchEmbedding(nn.Module):

    def __init__(self):
        super(PatchEmbedding, self).__init__()

        self.embedding_d = Config.embedding_d
        self.embedding_layer_input_shape = (Config.height, Config.width, 1)  # one line of 64 pixels -> a patch
        self.embedding_layer_output_shape = (Config.num_patches, Config.patch_size ** 2)

        self.conv = nn.Conv2d(in_channels=1,
                              out_channels=self.embedding_d,
                              kernel_size=Config.patch_size,
                              stride=Config.patch_size)

        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)

        return x.permute(0, 2, 1)


class MABlock(nn.Module):

    def __init__(self):
        super(MABlock, self).__init__()

        self.layer_norm = nn.LayerNorm(Config.embedding_d)
        self.multihead_attention = nn.MultiheadAttention(Config.embedding_d, Config.heads, Config.attn_dropout, )

    def forward(self, x):
        x = self.layer_norm(x)
        x, _ = self.multihead_attention(x, x, x)

        return x


class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()

        self.layer_norm = nn.LayerNorm(Config.embedding_d)
        self.mlp = nn.Sequential(
            nn.Linear(Config.embedding_d, Config.mlp_size),
            nn.GELU(),
            nn.Dropout(Config.dropout_rate),
            nn.Linear(Config.mlp_size, Config.embedding_d),
            nn.Dropout(Config.dropout_rate)
        )

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)

        return x


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.mablock = MABlock()
        self.mlp = MLP()

    def forward(self, x):
        x = self.mablock(x) + x
        x = self.mlp(x) + x

        return x
