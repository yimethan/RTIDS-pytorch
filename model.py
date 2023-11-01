import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import Config


class MultiheadAttention(nn.Module):

    def __init__(self):
        super(MultiheadAttention, self).__init__()

        self.heads = Config.heads
        self.d_model = Config.d_model
        self.dropout_rate = Config.dropout_rate
        self.d_k = self.d_model // self.heads

    def attention(self, q, k, v, mask=None):

        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = F.dropout(scores, p=self.dropout_rate)

        return torch.matmul(scores, v)

    def forward(self, q, k, v, mask=None):

        batch_size = Config.batch_size

        q = q.view(batch_size, -1, self.heads, self.d_k)
        k = k.view(batch_size, -1, self.heads, self.d_k)
        v = v.view(batch_size, -1, self.heads, self.d_k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = self.attention(q, k, v, self.d_k, mask)

        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return concat


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer).__init__()

        self.alpha = nn.Parameter(torch.ones(Config.d_model))
        self.eps = 1e-6
        self.beta = nn.Parameter(torch.zeros(Config.d_model))

        self.norm1 = self.calculate_norm(Config.d_model)
        self.norm2 = self.calculate_norm(Config.d_model)
        self.multihead_attention = MultiheadAttention()
        self.feed_forward = nn.Sequential(
            nn.Linear(Config.d_model, 1024),
            nn.ReLU(),
            nn.Dropout(Config.dropout_rate),
            nn.Linear(1024, Config.d_model)
        )
        self.dropout_1 = nn.Dropout(Config.dropout_rate)
        self.dropout_2 = nn.Dropout(Config.dropout_rate)

    def calculate_norm(self, x):

        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.beta
        return norm

    def forward(self, x, mask=None):
        x1 = self.norm1(x)
        x = self.multihead_attention(x1, x1, x1, mask)
        x = x + self.dropout_1(x)
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = x + self.dropout_2(x)

        return x


# https://github.com/mandersch/RTIDS/blob/main/RTIDS_Malte_Andersch.ipynb
class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()

        self.weights = nn.Parameter(torch.randn(78, Config.d_model))
        self.bias = nn.Parameter(torch.randn(78, Config.d_model))

    def forward(self, x):

        x = x.unsqueeze(-1)
        return x * self.weights + self.bias


class PositionalEncoding(nn.Module):
    def __init__(self):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(1000, Config.d_model)
        position = torch.arange(0, 1000, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, Config.d_model, 2).float() * (-np.log(10000.0) / Config.d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * np.sqrt(Config.d_model)
        x = x + self.pe[:x.size(0), :]
        return x


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.embedding = Embedding()
        self.positional_encoding = PositionalEncoding()
        self.encoder_layers = nn.ModuleList([EncoderLayer() for _ in range(6)])

    def forward(self, x, mask=None):

        x = self.embedding(x)
        x = self.positional_encoding(x)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
        x = self.norm(x)

        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder).__init__()
