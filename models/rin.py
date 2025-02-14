import torch
from torch import nn
from typing import Optional
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from utils.pos_embed import LearnableAxialPosEmb, SinusoidalPosEmb

from timm.models.vision_transformer import LayerScale

class Mlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            condition_dim=None,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)
        
        self.condition = None
        if condition_dim is not None:
            self.condition = nn.Sequential(
                nn.SiLU(),
                nn.Linear(condition_dim, in_features * 2),
            )
            self.condition[-1].weight.data.zero_()
            self.condition[-1].bias.data.zero_()
            self.condition_norm = norm_layer(in_features)

    def forward(self, x, condition = None):
        if condition is not None and self.condition is not None:
            condition = self.condition(condition).unsqueeze(1)
            shift, scale = condition.chunk(2, dim=-1)
            x = self.condition_norm(x) * (scale + 1) + shift
        
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Attention(nn.Module):
    def __init__(self,
                 dim: int,
                 kv_dim: Optional[int] = None,
                 num_heads: int = 1,
                 condition_dim: Optional[int] = None):
        super().__init__()
        
        self.q_norm = nn.LayerNorm(dim)
        self.k_norm = nn.LayerNorm(kv_dim if kv_dim is not None else dim)
        self.att = nn.MultiheadAttention(embed_dim=dim,
                                         kdim=kv_dim,
                                         vdim=kv_dim,
                                         num_heads=num_heads,
                                         batch_first=True)
        self.condition = None
        if condition_dim is not None:
            self.condition = nn.Sequential(
                nn.SiLU(),
                nn.Linear(condition_dim, dim * 2),
            )
            self.condition[-1].weight.data.zero_() # Initialize to zero
            self.condition[-1].bias.data.zero_() # Initialize to zero
            
    
    def forward(self,
                q: torch.Tensor,
                kv: Optional[torch.Tensor] = None,
                condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        q = self.q_norm(q)
        k = self.k_norm(kv) if kv is not None else q
        v = kv if kv is not None else q
        if condition is not None and self.condition is not None:
            condition = self.condition(condition).unsqueeze(1)
            shift, scale = condition.chunk(2, dim=-1)
            q = q * (scale + 1) + shift
        
        out = self.att(q, k, v, need_weights=False)[0]
        return out
        

class Layer(nn.Module):
    def __init__(self,
                 channels: int,
                 expansion: int,
                 num_heads: int,
                 time_condition_dim: Optional[int] = None,
                 init_values: float = 1e-3):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.scale_1 = LayerScale(channels, init_values=init_values)
        self.scale_2 = LayerScale(channels, init_values=init_values)
        self.ffn = Mlp(in_features=channels, hidden_features=channels * expansion, condition_dim=time_condition_dim)       
        self.attn = Attention(channels, num_heads=num_heads, condition_dim=time_condition_dim)

    def forward(self, z: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass of the block.

        Args:
            z (torch.Tensor): Tokenized input tensor of shape (B, N, C).
            t (torch.Tensor, optional): Time condition tensor of shape (B, T). Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (B, N, C).
        """
        
        z = z + self.scale_1(self.attn(q=z, condition=t))
        z = z + self.scale_2(self.ffn(self.norm(z), condition=t))
        return z

class Block(nn.Module):
    
    def __init__(self,
                 latent_dim: int,
                 interface_dim: int,
                 num_layers: int,
                 num_heads: int = 16,
                 expansion: int = 4,
                 time_cond_dim: Optional[int] = None,
                 conv: bool = False,
                 init_values: float = 1e-3):
        super().__init__()
        
        self.conv = self.conv = nn.Conv2d(interface_dim, interface_dim, 3, padding=1) if conv else None           
        
        self.read_norm = nn.LayerNorm(latent_dim)
        self.read_scale_1 = LayerScale(latent_dim, init_values=init_values)
        self.read_scale_2 = LayerScale(latent_dim, init_values=init_values)
        self.read_attn = Attention(
            dim=latent_dim,
            kv_dim=interface_dim,
            condition_dim=time_cond_dim,
            num_heads=num_heads,
        )
        self.read_ffn = Mlp(in_features=latent_dim, hidden_features=latent_dim * expansion, condition_dim=time_cond_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(Layer(latent_dim, num_heads=num_heads, expansion=expansion, time_condition_dim=time_cond_dim))
        
        self.write_norm = nn.LayerNorm(interface_dim)
        self.write_scale_1 = LayerScale(interface_dim, init_values=init_values)
        self.write_scale_2 = LayerScale(interface_dim, init_values=init_values)
        self.write_attn = Attention(
            dim=interface_dim,
            kv_dim=latent_dim,
            condition_dim=time_cond_dim,
            num_heads=num_heads,
        )
        self.write_ffn = Mlp(in_features=interface_dim, hidden_features=interface_dim * expansion, condition_dim=time_cond_dim)

    
    def forward(self, z: torch.Tensor, x: torch.Tensor, t: Optional[torch.Tensor] = None, ph: int = None, pw: int = None) -> tuple[torch.Tensor, torch.Tensor]:       
        if self.conv is not None and ph is not None and pw is not None:
            x = rearrange(x, "b (h w) c -> b c h w", h=ph, w=pw)
            x = self.conv(x)
            x = rearrange(x, "b c h w -> b (h w) c")
        
        z = z + self.read_scale_1(self.read_attn(q=z, kv=x, condition=t))
        z = z + self.read_scale_2(self.read_ffn(self.read_norm(z), condition=t))

        for layer in self.layers:
            z = layer(z, t)
        
        x = x + self.write_scale_1(self.write_attn(q=x, kv=z, condition=t))
        x = x + self.write_scale_2(self.write_ffn(self.write_norm(x), condition=t))
        return z, x

class RIN(nn.Module):
    def __init__(self,
                 patch_size: int,
                 num_latents: int,
                 latent_dim: int,
                 interface_dim: int,
                 time_cond_dim: int,
                 num_blocks: int,
                 num_layers_per_block: int,
                 num_heads: int = 16,
                 expansion: int = 4,
                 conv: bool = False,
                 global_skip: bool = False,
                 z_init_scale: float = 0.02):
        super().__init__()
        
        self.time_cond_mlp = nn.Sequential(
            SinusoidalPosEmb(dim=time_cond_dim),
            nn.Linear(time_cond_dim, time_cond_dim),
            nn.GELU(),
            nn.Linear(time_cond_dim, time_cond_dim)
        )
        
        self.patch_size = patch_size
        self.patch_embed = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (c p1 p2)", p1=patch_size, p2=patch_size
            ),
            nn.Linear(3 * patch_size ** 2, interface_dim)
        )
        
        self.pos_embed = LearnableAxialPosEmb(interface_dim)
        
        self.num_latents = num_latents
        self.latent_dim = latent_dim
        self.interface_dim = interface_dim
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                Block(latent_dim=latent_dim,
                      interface_dim=interface_dim,
                      num_layers=num_layers_per_block,
                      num_heads=num_heads,
                      expansion=expansion,
                      time_cond_dim=time_cond_dim,
                      conv=conv)
            )
        
        # Self-conditioning
        self.sc_mlp = Mlp(in_features=latent_dim, hidden_features=latent_dim * expansion, condition_dim=time_cond_dim)
        self.sc_norm = nn.LayerNorm(latent_dim)
        self.sc_norm.weight.data.zero_() # Initialize to zero
        
        # Readout
        self.readout = nn.Sequential(
            nn.LayerNorm(interface_dim),
            nn.Linear(interface_dim, 3 * patch_size ** 2),
        )
        
        self.z_init = nn.Parameter(torch.randn(num_latents, latent_dim))
        torch.nn.init.normal_(self.z_init, mean=0, std=z_init_scale)
        
        self.global_skip = global_skip

    def forward(self, x: tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        x, timesteps, prev_latents = x
        bsz, _, h, w = x.shape
        if self.global_skip:
            identity = x
        
        # Initialize latents
        z = repeat(self.z_init, 'n d -> b n d', b=bsz).to(x.device)
        
        t = self.time_cond_mlp(timesteps)
        
        # Latent self-conditioning
        if prev_latents is None:
            prev_latents = torch.zeros_like(z)
            prev_latents, _ = self.forward((x, timesteps, prev_latents))
            prev_latents = prev_latents.detach() # Detach from computation graph
        prev_latents = prev_latents + self.sc_mlp(prev_latents, condition=t)
        z = z + self.sc_norm(prev_latents)
        
        
        # Patch embed 
        pe_h, pe_w = h // self.patch_size, w // self.patch_size
        x = self.patch_embed(x)
        
        # Positional encoding
        pe = self.pos_embed(pe_h, pe_w, device=x.device)
        x = x + rearrange(pe, 'h w c -> 1 (h w) c')
        
        # Compute
        for block in self.blocks:
            z, x = block(z, x, t, ph=pe_h, pw=pe_w)
        
        # Readout
        x = self.readout(x)
        x = rearrange(x, "b (h w) (c p1 p2) -> b c (h p1) (w p2)", h=pe_h, w=pe_w, p1=self.patch_size, p2=self.patch_size)
        return z, x if not self.global_skip else x + identity

if __name__ == '__main__':
    
    model = RIN(
        patch_size=4,
        num_latents=256,
        latent_dim=128,
        interface_dim=256,
        time_cond_dim=64,
        num_blocks=6,
        num_layers_per_block=4,
        num_heads=2,
        expansion=4,
        conv=True,
        global_skip=True
    )
    
    x = torch.randn(5, 3, 128, 128)
    timesteps = torch.linspace(0, 1, 5).reshape(-1)
    out = model((x, timesteps, None))
    print(out[0].shape, out[1].shape)