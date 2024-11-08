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

    progress = '[' + '>' + 20*' ' + ']'

    t = time.time()
    out = IPy_display(IPy_Pretty(f'{progress} 0/{total_steps}'), display_id=True)
    losses = []

    for i in range(start, end, batch_size):
        loss = func(i, min(end, i + batch_size)).item()
        losses.append(loss)

        step = (i - start)//batch_size + 1
        t_left = (total_steps - step)*(time.time() - t)/step
        t_left = str(timedelta(seconds=t_left))[:-7]

        k = int(20*step/total_steps)
        progress = '[' + '='*k + '>' + ' '*(20 - k) + ']'

        out.update(IPy_Pretty(f'{progress} {step}/{total_steps}, {mean(losses):.4f} {t_left}'))

    out.update(IPy_Pretty(f'{progress} {total_steps}/{total_steps}, {mean(losses):.4f}'))

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
        if len(self.sin) < x.shape[2]:
            self.sin, self.cos = self.make_sin_cos(x.shape[2])
        x_ = x.reshape(x.shape[:-1] + (x.shape[-1]//2, 2))
        x_ = torch.stack([-x_[..., 1], x_[..., 0]], dim=-1).reshape(x.shape)
        x = x*self.cos[:x.shape[2]] + x_*self.sin[:x.shape[2]]
        return x

    def forward(self, q, k=None, v=None, causal=True):
        qkv = {'q':q}

        if k is None:
            qkv['k'] = q
        else:
            qkv['k'] = k
        
        if v is None:
            qkv['v'] = qkv['k']
        else:
            qkv['v'] = v

        for name in 'qkv':
            qkv[name] = self.qkv[name](qkv[name])
            qkv[name] = rearrange(qkv[name], 'b n (h a) -> b h n a', h=self.num_heads)

        qkv['q'], qkv['k'] = self.apply_rope(qkv['q']), self.apply_rope(qkv['k'])

        out = F.scaled_dot_product_attention(
            query=qkv['q'], key=qkv['k'], value=qkv['v'], is_causal=causal)
        
        out = rearrange(out, 'b h n a -> b n (h a)', h=self.num_heads)
        
        return self.o(out)
        
def make_sinusoidal(n, d, base=1e4):
    '''
    Makes sinusoidal positional embeddings of shape (n, d).
    '''
    p = torch.pow(base, -torch.linspace(0.0, 1.0, d//2))
    p = torch.arange(n, dtype=torch.float32).unsqueeze(1)*p.unsqueeze(0)
    p = torch.stack([torch.sin(p), torch.cos(p)], dim=-1).reshape(n, d)
    return p
