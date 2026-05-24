# -*-coding:utf-8 -*-

# File       : model.py
# Author     : hingmauc
# Time       : 2025/06/01 18:40
# Description：

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange, repeat

#忽略警告
import warnings
warnings.filterwarnings("ignore")

device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
 

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + \
                    torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


import numbers
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x

import math
class Heat2D(nn.Module):
    def __init__(self, infer_mode=False, res=14, dim=96, hidden_dim=96, **kwargs):
        super().__init__()
        self.res = res
        self.dwconv = nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.infer_mode = infer_mode
        self.to_k = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
        )

    def infer_init_heat2d(self, freq):
        weight_exp = self.get_decay_map((self.res, self.res), device=freq.device)
        self.k_exp = nn.Parameter(torch.pow(weight_exp[:, :, None], self.to_k(freq)), requires_grad=False)
        del self.to_k

    @staticmethod
    def get_cos_map(N=224, device=torch.device("cpu"), dtype=torch.float):
        weight_x = (torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(1, -1) + 0.5) / N
        weight_n = torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(-1, 1)
        weight = torch.cos(weight_n * weight_x * torch.pi) * math.sqrt(2 / N)
        weight[0, :] = weight[0, :] / math.sqrt(2)
        return weight

    @staticmethod
    def get_decay_map(resolution=(224, 224), device=torch.device("cpu"), dtype=torch.float):
        resh, resw = resolution
        weight_n = torch.linspace(0, torch.pi, resh + 1, device=device, dtype=dtype)[:resh].view(-1, 1)
        weight_m = torch.linspace(0, torch.pi, resw + 1, device=device, dtype=dtype)[:resw].view(1, -1)
        weight = torch.pow(weight_n, 2) + torch.pow(weight_m, 2)
        weight = torch.exp(-weight)
        return weight

    def forward(self, x: torch.Tensor, freq_embed=None):
        B, C, H, W = x.shape
        x = self.dwconv(x)

        x = self.linear(x.permute(0, 2, 3, 1).contiguous())
        x, z = x.chunk(chunks=2, dim=-1)

        if ((H, W) == getattr(self, "__RES__", (0, 0))) and (getattr(self, "__WEIGHT_COSN__", None).device == x.device):
            weight_cosn = getattr(self, "__WEIGHT_COSN__", None)
            weight_cosm = getattr(self, "__WEIGHT_COSM__", None)
            weight_exp = getattr(self, "__WEIGHT_EXP__", None)
            assert weight_cosn is not None
            assert weight_cosm is not None
            assert weight_exp is not None
        else:
            weight_cosn = self.get_cos_map(H, device=x.device).detach_()
            weight_cosm = self.get_cos_map(W, device=x.device).detach_()
            weight_exp = self.get_decay_map((H, W), device=x.device).detach_()
            setattr(self, "__RES__", (H, W))
            setattr(self, "__WEIGHT_COSN__", weight_cosn)
            setattr(self, "__WEIGHT_COSM__", weight_cosm)
            setattr(self, "__WEIGHT_EXP__", weight_exp)

        N, M = weight_cosn.shape[0], weight_cosm.shape[0]

        x = F.conv1d(x.contiguous().view(B, H, -1), weight_cosn.contiguous().view(N, H, 1))
        x = F.conv1d(x.contiguous().view(-1, W, C), weight_cosm.contiguous().view(M, W, 1)).contiguous().view(B, N, M, -1)

        if self.infer_mode:
            if not hasattr(self, "k_exp") or self.k_exp.shape[:2] != (H, W):
                weight_exp = self.get_decay_map((H, W), device=x.device)
                freq_resize = F.interpolate(
                    freq_embed.permute(2, 0, 1).unsqueeze(0),
                    size=(H, W), mode="bilinear", align_corners=False
                ).squeeze(0).permute(1, 2, 0).contiguous()
                with torch.no_grad():
                    self.k_exp = torch.pow(weight_exp[:, :, None], self.to_k(freq_resize)).detach()
            x = torch.einsum("bnmc,nmc->bnmc", x, self.k_exp)
        else:
            weight_exp = torch.pow(weight_exp[:, :, None], self.to_k(freq_embed))
            x = torch.einsum("bnmc,nmc -> bnmc", x, weight_exp)

        x = F.conv1d(x.contiguous().view(B, N, -1), weight_cosn.t().contiguous().view(H, N, 1))
        x = F.conv1d(x.contiguous().view(-1, M, C), weight_cosm.t().contiguous().view(W, M, 1)).contiguous().view(B, H, W, -1)
        x = self.out_norm(x)

        x = x * nn.functional.silu(z)
        x = self.out_linear(x)

        x = x.permute(0, 3, 1, 2).contiguous()

        return x


class ECA(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv     = nn.Conv1d(1, 1, kernel_size=k_size,
                                  padding=(k_size - 1) // 2, bias=False)

        self.sigmoid  = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)
        return x * y


class FMFN(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU, drop=0., use_eca=False):
        super(FMFN, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.ca = ECA(dim, k_size=3)

        self.project_in = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim,  kernel_size=3,
                                stride=1, padding=1,groups=hidden_dim,bias=False)
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.ca(x)
        x = self.project_in(x)
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.dwconv(x1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class HeatAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, heat_res):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1,
            groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.heat2d = Heat2D(infer_mode=False, res=heat_res, dim=dim, hidden_dim=dim)
        H, W = (heat_res, heat_res) if isinstance(heat_res, int) else heat_res
        self.freq_embed = nn.Parameter(torch.zeros(H, W, dim), requires_grad=True)
        nn.init.trunc_normal_(self.freq_embed, std=0.02)

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w
        c_per_head = c // self.num_heads

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn_logits = (q @ k.transpose(-2, -1)) * self.temperature

        T_feat = self.heat2d(x, freq_embed=self.freq_embed)
        T_chan = T_feat.mean(dim=(2, 3), keepdim=True)
        T_head = T_chan.view(b, self.num_heads, c_per_head, 1)

        attn_logits = attn_logits * T_head

        attn = attn_logits.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class TransformerBlock_Heat(nn.Module):
    def __init__(self,
                 dim,
                 heat_res,
                 num_heads=8,
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias'):
        super().__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = HeatAttention(dim, num_heads, bias, heat_res)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn   = FMFN(dim, int(dim * ffn_expansion_factor))

    def forward(self, x):
        y = self.attn(self.norm1(x))
        x = x + y
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


class Encoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 heat_res=(128, 128),
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):
        super(Encoder, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.ast_layer = nn.Sequential(
            *[TransformerBlock_Heat(dim=dim, num_heads=heads[0], heat_res=heat_res, ffn_expansion_factor=ffn_expansion_factor,
                                    bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        feature = self.ast_layer(inp_enc_level1)

        return feature


class Decoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 heat_res=(128, 128),
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(Decoder, self).__init__()

        self.encoder_level2 = nn.Sequential(
            *[TransformerBlock_Heat(dim=dim, num_heads=heads[0],heat_res=heat_res, ffn_expansion_factor=ffn_expansion_factor,
                                    bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.output = nn.Sequential(
            nn.Conv2d(int(dim), int(dim) // 2, kernel_size=3,
                      stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(int(dim) // 2, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias), )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp_img, feature):
        out_enc_level0 = feature
        out_enc_level1 = self.encoder_level2(out_enc_level0)
        if inp_img is not None:
            out_enc_level1 = self.output(out_enc_level1) + inp_img
        else:
            out_enc_level1 = self.output(out_enc_level1)
        return self.sigmoid(out_enc_level1)


class IterativeFusion(nn.Module):
    def __init__(self, in_channels,
                 depth=3,
                 heat_res=(128, 128),
                 num_heads = [8, 4, 2, 1],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias'
                 ):
        super(IterativeFusion, self).__init__()
        self.depth = depth
        self.vi_enhance = TransformerBlock_Heat(dim=in_channels, num_heads=num_heads[0],
                                                         heat_res=heat_res, ffn_expansion_factor=ffn_expansion_factor,
                                                         bias=bias, LayerNorm_type=LayerNorm_type).to(device)
        self.ir_enhance = TransformerBlock_Heat(dim=in_channels, num_heads=num_heads[0],
                                                         heat_res=heat_res, ffn_expansion_factor=ffn_expansion_factor,
                                                         bias=bias, LayerNorm_type=LayerNorm_type).to(device)

        self.fuse_block = TransformerBlock_Heat(dim=in_channels, num_heads=num_heads[0],
                                       heat_res=heat_res, ffn_expansion_factor=ffn_expansion_factor,
                                       bias=bias, LayerNorm_type=LayerNorm_type).to(device)

    def forward(self, feature_vi, feature_ir):
        for i in range(self.depth):
            feature_vi = self.vi_enhance(feature_vi)
            feature_ir = self.ir_enhance(feature_ir)
            fuse_feature = feature_vi + feature_ir
            fused_feature = self.fuse_block(fuse_feature)

        return fused_feature



if __name__ == '__main__':
    B, C, H, W = 3, 32, 128, 128
    num_heads = 8
    ffn_expansion_factor = 2
    bias = False
    LayerNorm_type = 'BiasFree'
    heat_res = (128, 128)

    transformer_block = TransformerBlock_Heat(C, heat_res, num_heads, ffn_expansion_factor, bias, LayerNorm_type)
    input_tensor = torch.randn(B, C, H, W)

    output_tensor = transformer_block(input_tensor)

    print("Input shape: ", input_tensor.shape)
    print("Output shape: ", output_tensor.shape)