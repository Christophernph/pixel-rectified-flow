import math

import einx
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

class PositionalEncoding2D(nn.Module):
    def __init__(self,
                 dim: int,
                 dtype_override=None):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        :param dtype_override: If set, overrides the dtype of the output embedding.
        """
        super().__init__()

        dim = int(np.ceil(dim / 4) * 2)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)
        self.dtype_override = dtype_override
        self.dim = dim
    
    def __validate_shape(self,
                         shape: torch.Size,
                         height: int,
                         width: int,
                         channels_last: bool,
                         add_batch_dim: bool) -> bool:
        if add_batch_dim:
            if len(shape) != 4:
                return False
        
        if channels_last:
            if shape[-3] != height or shape[-2] != width:
                return False
        else:
            if shape[-2] != height or shape[-1] != width:
                return False
        
        return True

    def forward(self,
                height: int,
                width: int,
                device: torch.device = None,
                dtype: torch.dtype = None,
                channels_last: bool = False,
                add_batch_dim: bool = False) -> torch.Tensor:
        if self.cached_penc is not None and self.__validate_shape(self.cached_penc.shape,
                                                                  height,
                                                                  width,
                                                                  channels_last,
                                                                  add_batch_dim):
            return self.cached_penc

        self.cached_penc = None
        pos_x = torch.arange(height, device=device, dtype=self.inv_freq.dtype)
        pos_y = torch.arange(width, device=device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq.to(device))
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq.to(device))
        emb_x = get_emb(sin_inp_x).unsqueeze(1)
        emb_y = get_emb(sin_inp_y)
        emb = torch.zeros(
            (height, width, self.dim * 2),
            device=device,
            dtype=(
                self.dtype_override if self.dtype_override is not None else dtype
            ),
        )
        emb[:, :, : self.dim] = emb_x
        emb[:, :, self.dim : 2 * self.dim] = emb_y

        if not channels_last:
            emb = rearrange(emb, "h w c -> c h w")

        if add_batch_dim:
            emb = emb.unsqueeze(0)

        self.cached_penc = emb
        return self.cached_penc
        
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = einx.multiply('i, j -> i j', x, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class LearnableAxialPosEmb(nn.Module):
    def __init__(self,
                 dim: int,
                 depth: int = 2,
                 activation = nn.SiLU()):
        super().__init__()
        
        self.vembedder = nn.Sequential()
        self.hembedder = nn.Sequential()
        for i in range(depth):
            in_dim = 1 if i == 0 else dim
            self.vembedder.append(nn.Linear(in_dim, dim))
            self.hembedder.append(nn.Linear(in_dim, dim))
            if i != depth - 1:
                self.vembedder.append(activation)        
                self.hembedder.append(activation)

                
    def forward(self, height: int, width: int, device: torch.device = None):

        vranges = torch.linspace(-1, 1, height, device=device).unsqueeze(-1)
        hrange = torch.linspace(-1, 1, width, device=device).unsqueeze(-1)
        
        vemb = rearrange(self.vembedder(vranges), "j d -> j 1 d")
        hemb = rearrange(self.hembedder(hrange), "i d -> 1 i d")
        return hemb + vemb

if __name__ == '__main__':
    model = LearnableAxialPosEmb(64).to('cuda:0')
    print(model(256, 256, device='cuda:0').shape)