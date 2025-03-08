import math

import torch
import torch.nn.functional as F
from torch import nn

from .multihead_attention import MultiheadAttention
from .position_embedding import SinusoidalPositionalEmbedding


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)

def buffered_future_mask(tensor, tensor2=None):
    dim1 = dim2 = tensor.size(0)
    if tensor2 is not None:
        dim2 = tensor2.size(0)
    future_mask = torch.triu(fill_with_neg_inf(torch.ones(dim1, dim2)), 1+abs(dim2-dim1))
    if tensor.is_cuda:
        future_mask = future_mask.to(tensor.device)
    return future_mask[:dim1, :dim2]

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m

def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m

class ACCT(nn.Module):
    def __init__(self, embed_dim, num_heads, layers, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False, position_embedding=False):
        super().__init__()
        self.dropout = embed_dropout  # Embedding dropout

        self.embed_scale = math.sqrt(embed_dim)
        if position_embedding:
            self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
        else:
            self.embed_positions = None

        self.encoder = ACCTEncoder(layers, embed_dim,
                                num_heads=num_heads,
                                attn_dropout=attn_dropout,
                                relu_dropout=relu_dropout,
                                res_dropout=res_dropout,
                                attn_mask=attn_mask)

        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = True
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def forward(self, x_in, x_in_k, x_in_v, text):
        '''
        x_in:目标模态输入
        x_in_k、x_in_v:源模态输入（两个输入其实完全相同）
        text：文本模态输入
        '''
        x = self.embed_scale * x_in
        if self.embed_positions is not None:
            x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)  # Add positional embedding
        x = F.dropout(x, p=self.dropout, training=self.training)

        x_k = self.embed_scale * x_in_k
        x_v = self.embed_scale * x_in_v
        text_k = self.embed_scale * text
        text_v = self.embed_scale * text
        x_kv = self.embed_scale * x_in_v
        if self.embed_positions is not None:
            x_k += self.embed_positions(x_in_k.transpose(0, 1)[:, :, 0]).transpose(0, 1)  # Add positional embedding
            x_v += self.embed_positions(x_in_v.transpose(0, 1)[:, :, 0]).transpose(0, 1)
            x_kv += self.embed_positions(x_in_v.transpose(0, 1)[:, :, 0]).transpose(0, 1)# Add positional embedding
            text_v += self.embed_positions(text.transpose(0, 1)[:, :, 0]).transpose(0, 1)
            text_k += self.embed_positions(text.transpose(0, 1)[:, :, 0]).transpose(0, 1)
        x_k = F.dropout(x_k, p=self.dropout, training=self.training)
        x_v = F.dropout(x_v, p=self.dropout, training=self.training)
        x_kv = F.dropout(x_kv, p=self.dropout, training=self.training)
        text_v = F.dropout(text_v, p=self.dropout, training=self.training)
        text_k = F.dropout(text_k, p=self.dropout, training=self.training)

        # encoder layers
        intermediates = [x]

        x = self.encoder(x, x_k, x_v, x_kv, text_k, text_v)
        intermediates.append(x)

        if self.normalize:
            x = self.layer_norm(x)

        return x

class ACCTEncoder(nn.Module):
    def __init__(self, attn_depth, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1, sigma=0.5,
                 attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.attn_mask = attn_mask
        self.sigmoid = nn.Sigmoid()

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        self.fc1 = Linear(self.embed_dim, 4 * self.embed_dim)  # The "Add & Norm" part in the paper
        self.fc2 = Linear(4 * self.embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])

        self.layers1 = nn.ModuleList([])
        self.layers2 = nn.ModuleList([])

        for _ in range(attn_depth):
            self.layers1.append(MultiheadAttention(embed_dim=self.embed_dim,num_heads=self.num_heads,attn_dropout=attn_dropout))

        for _ in range(attn_depth):
            self.layers2.append(MultiheadAttention(embed_dim=self.embed_dim,num_heads=self.num_heads,attn_dropout=attn_dropout))

        self.score = nn.Linear(2 * self.embed_dim, self.embed_dim)

        self.sigma = sigma

    def forward(self, x, x_k, x_v, x_kv, text_k, text_v):

        len, b, dim = x.shape
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        mask = buffered_future_mask(x, x_k) if self.attn_mask else None

        x_k = self.maybe_layer_norm(0, x_k, before=True)
        x_v = self.maybe_layer_norm(0, x_v, before=True)
        x_kv = self.maybe_layer_norm(0, x_kv, before=True)
        mask2 = buffered_future_mask(x_kv, text_k) if self.attn_mask else None
        text_k = self.maybe_layer_norm(0, text_k, before=True)
        text_v = self.maybe_layer_norm(0, text_v, before=True)
        for attn in self.layers1:
            x, _ = attn(query=x, key=x_k, value=x_v, attn_mask=mask)

        for attn in self.layers2:
            source_x, _ = attn(query=x_kv, key=text_k, value=text_v, attn_mask=mask2)

        score = self.sigmoid(self.score(torch.cat((x, source_x), dim=2)))
        zeros = torch.zeros(len, b, dim).to(x.device)
        zeros = zeros + self.sigma
        score = torch.where(score <= self.sigma, zeros, score)

        x = score * x + (1 - score) * source_x

        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x
