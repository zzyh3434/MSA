import torch
from torch import nn
import math

class RE(nn.Module):
    def __init__(self, my_dim, miu):
        super().__init__()

        self.adaptive_vat1 = nn.Linear(1 * my_dim, int(my_dim / 2))
        self.adaptive_tva1 = nn.Linear(1 * my_dim, int(my_dim / 2))
        self.adaptive_vat2 = nn.Linear(int(my_dim / 2), my_dim)
        self.adaptive_tva2 = nn.Linear(int(my_dim / 2), my_dim)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.miu = miu

    def forward(self, vat, tva):

        b, len, dim = vat.shape

        adaptive_vat = self.sigmoid(self.adaptive_vat2(self.relu(self.adaptive_vat1(vat))))
        adaptive_tva = self.sigmoid(self.adaptive_tva2(self.relu(self.adaptive_tva1(tva))))
        adaptive_vat2 = adaptive_vat
        adaptive_tva2 = adaptive_tva

        zeros = torch.zeros(b, len, dim).to(text_h.device)
        ones = torch.ones(b, len, dim).to(text_h.device)
        adaptive_vat = torch.where(adaptive_vat <= self.miu, zeros, ones)
        adaptive_tva = torch.where(adaptive_tva <= self.miu, zeros, ones)

        vat2 = vat * adaptive_vat2 + tva * adaptive_tva * (1 - adaptive_vat2)
        tva2 = tva * adaptive_tva2 + vat * adaptive_vat * (1 - adaptive_tva2)

        return vat2, tva2