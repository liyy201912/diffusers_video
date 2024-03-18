from typing import Optional, Union, List

from functools import partial

from einops import rearrange, pack, unpack

import torch
from torch import nn

from ...utils import deprecate, is_torch_version, logging
from ...utils.torch_utils import apply_freeu
from ..attention import Attention



# Lumiere spatio-temporal blocks
# We assume all features are with following dimensions [batch * time, channels, height, width]
# thus we need to fix time (frames) dimensions in module __init__

# from lucidrains
def compact_dict(**kwargs):
    return {k: v for k, v in kwargs.items() if v is not None}

# from lucidrains
def _init_bilinear_kernel_1d(conv: nn.Module):
    nn.init.zeros_(conv.weight)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)

    channels = conv.weight.shape[0]
    bilinear_kernel = torch.Tensor([0.5, 1., 0.5])
    diag_mask = torch.eye(channels).bool()
    conv.weight.data[diag_mask] = bilinear_kernel


class LumiereBase(nn.Module):
    """
    Base model for Lumiere block
    """
    def __init__(
            self,
            time_dim: Optional[Union[int, bool]] = None
            ) -> None:
        super().__init__()
        self.time_dim = time_dim
        # self.register_buffer('time_dim', torch.tensor(time_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

class ConvInflationBlock(LumiereBase):
    """
    Lumiere conv inflation block. after each conv (block/layer)
    This basically is a (2+1)D convolution.
    """
    def __init__(
        self,
        channels: int,
        conv2d_kernel_size: int,
        conv1d_kernel_size: int,
        num_groups: int = 32,
        time_dim: Optional[Union[int, bool]] = None,
    ):
        super(ConvInflationBlock, self).__init__(time_dim=time_dim)

        group_norm = partial(nn.GroupNorm, num_groups=num_groups)

        # init spatial_conv (Conv2D + norm + activation) 
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=conv2d_kernel_size, padding='same'),
            group_norm(num_channels=channels),
            nn.SiLU(),  # inplace is not friendly for some reasons 
        )

        # init temporal_conv (Conv1D + norm + activation), we definitely need reshape :)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=conv1d_kernel_size, padding='same'),
            group_norm(num_channels=channels),
            nn.SiLU(),  # inplace is not friendly for some reasons 
        )

        # init linear projection, we use Conv1D such that no reshape needed :)
        self.proj_out = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1)


    def forward(self, x: torch.Tensor, batch_size: Optional[int]) -> torch.Tensor:
        
        batch_frames, channels, height, width = x.shape

        if batch_size is None:
            batch_size = batch_frames // self.time_dim
        
        num_frames = batch_frames // batch_size

        x = self.spatial_conv(x)

        # rearrange channels and timestamps to last two dims for temporal conv and proj_out
        # x = rearrange(x, '(b t) c h w -> b h w c t', **compact_dict(b=batch_size, t=self.time_dim))

        # x, ps = pack([x], '* c t')
        x = x[None, ...].reshape(batch_size, num_frames, channels, height, width).permute(0, 3, 4, 2, 1).reshape(-1, channels, num_frames)

        x = self.temporal_conv(x)
    
        x = self.proj_out(x)

        # x = unpack(x, ps, '* c t')[0]
        
        # x = rearrange(x, 'b h w c t -> (b t) c h w')

        x = x.reshape(batch_size, height, width, channels, num_frames).permute(0, 4, 3, 1, 2).flatten(0, 1)

        return x


class AttentionInflationBlock(LumiereBase):
    """
    Lumiere conv inflation block. after each conv (block/layer)
    """
    def __init__(self, 
                 dim: int, 
                 num_layers: int, 
                 norm_elementwise_affine: bool = True,
                 norm_eps: float = 1e-5,
                 attention_head_dim: int = 512,
                 upcast_attention: bool = False,
                 time_dim: Optional[Union[bool, int]] = None) -> None:
        super().__init__(time_dim)
        
        
        self.temporal_attn = nn.ModuleList()

        for _ in range(num_layers):
            attn = nn.Sequential(
                nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps),
                Attention(query_dim=dim,
                          heads=dim // attention_head_dim,
                          dim_head=attention_head_dim,
                          eps=1e-6,
                          upcast_attention=upcast_attention,
                          norm_num_groups=32,
                          bias=True,
                          residual_connection=True,
                          )
            )

            self.temporal_attn.append(attn)


        self.proj_out = nn.Linear(in_features=dim, out_features=dim)


    def forward(self, x: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
        
        # todo: check actual shape in Diffuser MidBlock
        assert x.dim() == 4, f"Expect x to be 4-D tensor but got {x.shape}"

        batch_frames, channels, height, width = x.shape

        if batch_size is None:
            batch_size = x.size(0) // self.time_dim

        num_frames = batch_frames // batch_size
        
        # rearrange that attention along time dimention
        # x = rearrange(x, '(b t) c h w -> b h w t c', **compact_dict(b=batch_size, t=self.time_dim))
        # pack b h w -> (b h w)
        # x, ps = pack([x], '* t c')
        x = x[None, ...].reshape(batch_size, num_frames, channels, height, width).permute(0, 3, 4, 1, 2).reshape(-1, num_frames, channels)

        # do attention
        for attn in self.temporal_attn:

            x = attn(x)

        x = self.proj_out(x)

        # unpack (b h w) -> b h w
        # x = unpack(x, ps, '* t c')[0]

        # rearrange back as input shape
        # x = rearrange(x, 'b h w t c -> (b t) c h w')
        x = x.reshape(batch_size, height, width, num_frames, channels).permute(0, 3, 4, 1, 2).flatten(0, 1)
        # residual connection as paper's Figure 4(c)
        return x


class TemporalDownsample(LumiereBase):
    """
    Temporal Downsample layer. After Spatial Downsample
    """
    def __init__(
            self, 
            dim,
            time_dim: Optional[Union[int, bool]] = None
            ) -> None:
        super().__init__(time_dim)

        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)
        
        _init_bilinear_kernel_1d(self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        assert x.dim() == 4, f"Expect x to be 4-D tensor but got {x.shape}"
        batch_size = x.size(0) // self.time_dim
        # rearrange channels and timestamps to last two dims for temperol downsample
        x = rearrange(x, '(b t) c h w -> b h w c t', **compact_dict(b=batch_size, t=self.time_dim))
        assert x.shape[-1] > 1, 'time dimension must be greater than 1 to be compressed'
        # pack to 3D tensor
        x, ps = pack([x], '* c t')
        x = self.conv(x)
        # unpack to 4D tensor
        x = unpack(x, ps, '* c t')[0]
        # rearrange back
        x = rearrange(x, 'b w h c t -> (b t) c h w')
        return x


class TemporalUpsample(LumiereBase):
    """
    Temporal Upsample layer. After Spatial Upsample
    """
    def __init__(
            self,
            dim,
            time_dim: Optional[Union[int, bool]] = None
            ) -> None:
        super().__init__(time_dim)

        self.conv = nn.ConvTranspose1d(dim, dim, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)
        _init_bilinear_kernel_1d(self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4, f"Expect x to be 4-D tensor but got {x.shape}"
        batch_size = x.size(0) // self.time_dim
        # rearrange channels and timestamps to last two dims for temperol downsample
        x = rearrange(x, '(b t) c h w -> b h w c t', **compact_dict(b=batch_size, t=self.time_dim))

        # pack to 3D tensor
        x, ps = pack([x], '* c t')
        x = self.conv(x)
        # unpack to 4D tensor
        x = unpack(x, ps, '* c t')[0]
        # rearrange back
        x = rearrange(x, 'b w h c t -> (b t) c h w')
        return x
    

# post module hook wrapper, we adopt the same method as lucidrains
class PostModuleHookWrapper(nn.Module):
    def __init__(self, temporal_module: nn.Module):
        super().__init__()
        self.temporal_module = temporal_module

    def forward(self, _, input, output):
        output = self.temporal_module(output)
        return output

def insert_temporal_modules_(modules: List[nn.Module], temporal_modules: nn.ModuleList):
    assert len(modules) == len(temporal_modules)

    for module, temporal_module in zip(modules, temporal_modules):
        module.register_forward_hook(PostModuleHookWrapper(temporal_module))