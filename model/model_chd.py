import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import Config


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.out = nn.Linear(Config.d_model * Config.features, Config.num_classes)

    def forward(self, x, mask=None):

        encoder_output = self.encoder(x, mask)
        decoder_output = self.decoder(x, encoder_output, mask)
        x = decoder_output.view(Config.batch_size, -1)
        # print('x.shape after x = decoder_output.view()', x.shape)  # batch_size, features * d_model
        x = self.out(x)
        # print('self.out(x)', x.shape)  # batch_size, num_classes
        x = F.softmax(x, dim=1)
        # print('F.softmax(x, dim=1)', x.shape)  # batch_size, num_classes

        return x


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

        scores = self.attention(q, k, v, mask)

        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return concat


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()

        self.alpha = nn.Parameter(torch.ones(Config.d_model))
        self.eps = 1e-6
        self.beta = nn.Parameter(torch.zeros(Config.d_model))

        self.norm1 = nn.LayerNorm(Config.d_model)
        self.norm2 = nn.LayerNorm(Config.d_model)
        self.multihead_attention = MultiheadAttention()
        self.ff = nn.Sequential(
            nn.Linear(Config.d_model, Config.mlp_size),
            nn.ReLU(),
            nn.Dropout(Config.dropout_rate),
            nn.Linear(Config.mlp_size, Config.d_model)
        )
        self.dropout_1 = nn.Dropout(Config.dropout_rate)
        self.dropout_2 = nn.Dropout(Config.dropout_rate)

    def forward(self, x, mask=None):
        # print('EncoderLayer input shape: ', x.shape)  # batch, 11, 32
        # shape same for every layers below
        x1 = self.norm1(x)
        x2 = self.multihead_attention(x1, x1, x1, mask)

        do1 = self.dropout_1(x2)

        # x = x.reshape(Config.batch_size, -1, Config.d_model)
        x = x + do1

        x = self.norm2(x)
        x1 = self.ff(x)

        do2 = self.dropout_2(x1)
        x = x + do2

        return x


# https://github.com/mandersch/RTIDS/blob/main/RTIDS_Malte_Andersch.ipynb
class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()

        self.weights = nn.Parameter(torch.randn(Config.batch_size, Config.features, Config.d_model))
        self.bias = nn.Parameter(torch.randn(Config.batch_size, Config.features, Config.d_model))

    def forward(self, x):
        # print('Embedding input shape: ', x.shape)  # batch, features
        x = x.unsqueeze(-1).view(Config.batch_size, -1, 1)
        # print('x.unsqueeze(-1).view(Config.batch_size, -1, 1)', x.shape)  # batch, 11, 1

        x = x * self.weights
        # print('x * self.weights', x.shape)  # batch, features, d_model
        x = x + self.bias
        # print('x + self.bias', x.shape)  # batch, features, d_model

        return x


class PositionalEncoding(nn.Module):
    def __init__(self):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(1000, Config.d_model)
        position = torch.arange(0, 1000, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, Config.d_model, 2).float() * (-np.log(10000.0) / Config.d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        # self.register_buffer('pe', pe)
        self.register_buffer('pe', pe[:, :Config.features, :])

    def forward(self, x):
        # print('--- Positional Encoding ---')
        x = x * np.sqrt(Config.d_model)
        # print('x * np.sqrt(Config.d_model)', x.shape)  # batch * features * d_model
        p = self.pe[:x.size(0), :]
        # print('self.pe[:x.size(0), :]', p.shape)  # batch * 1 * d_model
        x = x + p
        # print('x + p', x.shape)  # batch * features * d_model
        return x


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.embedding = Embedding()
        self.positional_encoding = PositionalEncoding()
        self.encoder_layers = nn.ModuleList([EncoderLayer() for _ in range(Config.layers)])
        self.norm = nn.LayerNorm(Config.d_model)

    def forward(self, x, mask=None):
        # print('--- Encoder ---')

        x = self.embedding(x)  # ([batch, features, d_model]) = ([batch, 11, 32])

        x = self.positional_encoding(x)
        # print('after positional encoding', x.shape)  # batch * 11 * 32

        for encoder_layer in self.encoder_layers:
            # print('--- Encoder Layer ---')
            x = encoder_layer(x, mask)

        x = self.norm(x)

        return x


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()

        self.norm1 = nn.LayerNorm(Config.d_model)
        self.norm2 = nn.LayerNorm(Config.d_model)
        self.norm3 = nn.LayerNorm(Config.d_model)
        self.mask_att = MultiheadAttention()
        self.att = MultiheadAttention()
        self.ff = nn.Sequential(
            nn.Linear(Config.d_model, Config.mlp_size),
            nn.ReLU(),
            nn.Dropout(Config.dropout_rate),
            nn.Linear(Config.mlp_size, Config.d_model)
        )
        self.dropout1 = nn.Dropout(Config.dropout_rate)
        self.dropout2 = nn.Dropout(Config.dropout_rate)
        self.dropout3 = nn.Dropout(Config.dropout_rate)

    def forward(self, x, encoder_output, mask):
        x1 = self.norm1(x)
        x2 = self.mask_att(x1, x1, x1, mask)
        x = x + self.dropout1(x2)
        x1 = self.norm2(x)
        x2 = self.att(x1, encoder_output, encoder_output)
        x = x + self.dropout2(x2)
        x1 = self.norm3(x)
        x2 = self.ff(x1)
        x = x + self.dropout3(x2)

        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.embed = Embedding()
        self.positional_encoding = PositionalEncoding()
        self.decoder_layers = nn.ModuleList([DecoderLayer() for _ in range(Config.layers)])
        self.norm = nn.LayerNorm(Config.d_model)

    def forward(self, x, encoder_output, mask=None):
        x = self.embed(x)
        x = self.positional_encoding(x)
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, encoder_output, mask)
        x = self.norm(x)

        return x
