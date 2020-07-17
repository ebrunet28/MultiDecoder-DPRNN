# Created on 2018/12/10
# Author: Kaituo XU
import math
import torch
def pad_segment(mixture_w):
    M, N, T = mixture_w.shape
    K, P = 10, 5
    S = math.ceil((T-K)/P) + 1
    print(S)
    pad_len = (S-1)*P + K - T
    if pad_len > 0:
        padding = torch.zeros((M, N, pad_len))
        mixture_w = torch.cat((mixture_w, padding), dim = 2)
    return mixture_w.unfold(-1, K, P).permute((0, 1, 3, 2))

mixture_w = torch.zeros((100, 20, 17))
print(pad_segment(mixture_w).shape)

