from numpy import ceil, mean, power
from datetime import timedelta
import time

from IPython.display import display as IPy_display
from IPython.display import Pretty as IPy_Pretty

import torch.nn.functional as F
from torch import nn
import torch

def nice_int_params(m):
    '''
    Returns a nicely formatted total number of parameters in an nn.Module.
    '''
    x = sum([var.numel() for var in m.parameters()])
    x = str(x)[::-1]
    x = ','.join([x[i:i + 3] for i in range(0, len(x), 3)])
    x = x[::-1]
    return x

def scale(x):
    '''
    Returns the reciprocal square root of a number as a float32 tensor.
    (Useful for scaling weights.)
    '''
    return torch.pow(torch.Tensor(x), -0.5)

def check(x, name=None):
    '''
    Get the basic information about a tensor.
    '''
    is_nan = torch.any(torch.isnan(x))
    mean = torch.mean(x)
    ma = x.max()
    mi = x.min()
    std = torch.std(x)
    A = [is_nan, mean, ma, mi, std]
    for i in range(5):
        A[i] = A[i].detach().cpu().numpy()
    is_nan, mean, ma, mi, std = A
    if name is None:
        name = ''
    else:
        name = name + ': '

    print(f'{name}Shape: {list(x.shape)}, Mean: {mean}, Dev: {std}, Max/Min: {ma}, {mi}, NaNs: {is_nan}')
    
def epoch_display(func, start, end, batch_size):
    '''
    A function which nicely displays epochs and manages batching the data.
    Given some function, it will pass two integers to it each step which
    are the start and end indices of the data within that batch. The args
    `start` and `end` are the overall start and end indices for the epoch.
    '''
    total_steps = int(ceil((end - start)/batch_size))

    t = time.time()
    out = IPy_display(IPy_Pretty('Starting ...'), display_id=True)
    losses = []

    for i in range(start, end, batch_size):
        loss = func(i, min(end, i + batch_size)).item()
        losses.append(loss)

        step = (i - start)//batch_size + 1
        t_left = (total_steps - step)*(time.time() - t)/step
        t_left = str(timedelta(seconds=t_left))[:-7]

        out.update(IPy_Pretty(f'{step}/{total_steps}, {mean(losses):.4f} {t_left}'))

    out.update(IPy_Pretty(f'{total_steps}/{total_steps}, {mean(losses):.4f}'))

    return losses

class SwiGLU(nn.Module):
    '''
    Implements SwiGLU.
    '''
    def __init__(self, width, scale_factor=8/3):
        super().__init__()
        self.large_width = int(width*scale_factor)

        self.up = nn.Linear(width, 2*self.large_width)
        self.down = nn.Linear(self.large_width, width)

    def forward(self, x):
        u, v = self.up(x).split(self.large_width, dim=-1)
        u = u*F.silu(v)
        v = self.down(u)
        return v

class MHA(nn.Module):
    '''
    Simple Multihead Attention.
    '''
    def __init__(self, width, num_heads):
        super().__init__()
        self.width = width
        self.num_heads = num_heads

        assert self.width % self.num_heads == 0

        self.q = nn.Linear(self.width, width, bias=False)
        self.kv = nn.Linear(self.width, 2*width, bias=False)
        self.mha = nn.MultiheadAttention(self.width, self.num_heads, batch_first=True, bias=False)
        self.o = nn.Linear(self.width, self.width, bias=False)

    def forward(self, x, y, x_mask, decoder):
        '''
        `x` gives keys and values and  `y` gives queries.
        `True` in  `x_mask` indicates that `y` should attend to `x` there.
        '''
        q = self.q(y)
        k, v = self.kv(x).split(self.width, dim=-1)

        if decoder:
            N, M = y.shape[1], x.shape[1]
            mask = torch.arange(N)[:, None] < torch.arange(M)[None, :]
        else:
            mask = None

        out = self.mha(
            q, k, v, need_weights=False, key_padding_mask=torch.logical_not(x_mask),
            attn_mask=mask, is_causal=decoder)[0]

        return self.o(out)

class Grokfast:
    '''
    Implementation of Grokfast, from "Lee et al. 2024" (arXiv:2405.20233).
    Uses a second beta instead of a lambda and bias corrects.
    Basically combines the gradients with momentum before it's even passed to the optimizer.
    In other words, with the second beta being 0.0, would do nothing, and with it being 1.0,
    effectively is just momentum, however can increase generalisation inbetween.
    Pass `params` in the same order every time.
    '''
    def __init__(self, params, betas=(0.99, 0.85)):
        self.m = [torch.zeros_like(i) for i in params],
        self.betas = betas
        self.k = 0
    
    def update(self, params):
        self.k += 1

        for j, para in enumerate(params):
            if para.grad is not None:
                g = para.grad.data.detach()
                self.m[j] = self.betas[0]*self.m[j] + (1 - self.betas[0])*g
                new_grad = g*(1 - self.betas[1])
                new_grad = new_grad + self.betas[1]*self.m[j]/(1.0 - power(self.betas[1], self.k))
                para.grad.data = new_grad

def make_sinusoidal(n, d, base=1e4):
    '''
    Makes sinusoidal positional embeddings of shape (n, d).
    '''
    p = torch.pow(base, -torch.linspace(0.0, 1.0, d//2))
    p = torch.arange(n, dtype=torch.float32).unsqueeze(1)*p.unsqueeze(0)
    p = torch.stack([torch.sin(p), torch.cos(p)], dim=-1).reshape(n, d)
    return p
