from numpy import ceil, mean, power
from datetime import timedelta
import time

from IPython.display import display as IPy_display
from IPython.display import Pretty as IPy_Pretty

import torch.nn.functional as F
from torch import nn
import torch

from einops import rearrange

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
    Simple Multihead Attention. Uses RoPE.
    '''
    def __init__(self, width, num_heads, max_length=None, att_width=None):
        super().__init__()
        self.width = width
        self.num_heads = num_heads
        self.max_length = max_length

        if att_width is None:
            self.att_width = self.width//self.num_heads
        else:
            self.att_width = att_width

        self.qkv = nn.ModuleDict(
            {a:nn.Linear(
                self.width, self.num_heads*self.att_width, bias=False
            ) for a in 'qkv'})
             
        self.mha = nn.MultiheadAttention(self.att_width*self.num_heads, self.num_heads, batch_first=True, bias=False)
        self.o = nn.Linear(self.num_heads*self.att_width, self.width, bias=False)
        
        if self.max_length is not None:
            self.sin, self.cos = self.make_sin_cos(self.max_length)
        else:
            self.sin, self.cos = [], []
    
    def make_sin_cos(self, n):
        theta = torch.arange(n).unsqueeze(-1)*torch.pow(
            1e4, torch.repeat_interleave(
                -torch.linspace(0.0, 1.0, self.att_width//2), 2))
        return torch.sin(theta), torch.cos(theta)
    
    def apply_rope(self, x):
        x = rearrange(x, 'b n (h a) -> b h n a', h=self.num_heads)
        if len(self.sin) < x.shape[2]:
            self.sin, self.cos = self.make_sin_cos(x.shape[2])
        x_ = x.reshape(x.shape[:-1] + (x.shape[-1]//2, 2))
        x_ = torch.stack([-x_[..., 1], x_[..., 0]], dim=-1).reshape(x.shape)
        x = x*self.cos[:x.shape[2]] + x_*self.sin[:x.shape[2]]
        x = rearrange(x, 'b h n a -> b n (h a)', h=self.num_heads)
        
        return x

    def forward(self, q, k=None, v=None, mask=None, decoder=True):
        '''
        `True` in  `mask` indicates that that entry to the K/V axis should be attended to.
        '''
        if k is None:
            k = q
        if v is None:
            v = k
        if mask is not None:
            mask = torch.logical_not(mask)
            
        q, k, v = self.qkv['q'](q), self.qkv['k'](k), self.qkv['v'](v)
        q, k = self.apply_rope(q), self.apply_rope(k)

        attn_mask = None
        if decoder:
            N, M = q.shape[1], k.shape[1]
            attn_mask = torch.arange(N)[:, None] < torch.arange(M)[None, :]

        out = self.mha(
            q, k, v, need_weights=False, key_padding_mask=mask,
            attn_mask=attn_mask, is_causal=decoder)[0]

        return self.o(out)
        
def make_sinusoidal(n, d, base=1e4):
    '''
    Makes sinusoidal positional embeddings of shape (n, d).
    '''
    p = torch.pow(base, -torch.linspace(0.0, 1.0, d//2))
    p = torch.arange(n, dtype=torch.float32).unsqueeze(1)*p.unsqueeze(0)
    p = torch.stack([torch.sin(p), torch.cos(p)], dim=-1).reshape(n, d)
    return p
