# © 2022. Triad National Security, LLC. All rights reserved.

# This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos

# National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.

# Department of Energy/National Nuclear Security Administration. All rights in the program are

# reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear

# Security Administration. The Government is granted for itself and others acting on its behalf a

# nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare

# derivative works, distribute copies to the public, perform publicly and display publicly, and to permit

# others to do so.

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil
from collections import OrderedDict

NORM_LAYERS = { 'bn': nn.BatchNorm2d, 'in': nn.InstanceNorm2d, 'ln': nn.LayerNorm }

# Replace the key names in the checkpoint in which legacy network building blocks are used 
def replace_legacy(old_dict):
    li = []
    for k, v in old_dict.items():
        k = (k.replace('Conv2DwithBN', 'layers')
              .replace('Conv2DwithBN_Tanh', 'layers')
              .replace('Deconv2DwithBN', 'layers')
              .replace('ResizeConv2DwithBN', 'layers'))
        li.append((k, v))
    return OrderedDict(li)

class Conv2DwithBN(nn.Module):
    def __init__(self, in_fea, out_fea, 
                kernel_size=3, stride=1, padding=1,
                bn=True, relu_slop=0.2, dropout=None):
        super(Conv2DwithBN,self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if bn:
            layers.append(nn.BatchNorm2d(num_features=out_fea))
        layers.append(nn.LeakyReLU(relu_slop, inplace=True))
        if dropout:
            layers.append(nn.Dropout2d(0.8))
        self.Conv2DwithBN = nn.Sequential(*layers)

    def forward(self, x):
        return self.Conv2DwithBN(x)

class ResizeConv2DwithBN(nn.Module):
    def __init__(self, in_fea, out_fea, scale_factor=2, mode='nearest'):
        super(ResizeConv2DwithBN, self).__init__()
        layers = [nn.Upsample(scale_factor=scale_factor, mode=mode)]
        layers.append(nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(num_features=out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.ResizeConv2DwithBN = nn.Sequential(*layers)

    def forward(self, x):
        return self.ResizeConv2DwithBN(x)
 
class Conv2DwithBN_Tanh(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1):
        super(Conv2DwithBN_Tanh, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        layers.append(nn.BatchNorm2d(num_features=out_fea))
        layers.append(nn.Tanh())
        self.Conv2DwithBN = nn.Sequential(*layers)

    def forward(self, x):
        return self.Conv2DwithBN(x)

class ConvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn', relu_slop=0.2, dropout=None):
        super(ConvBlock,self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(relu_slop, inplace=True))
        if dropout:
            layers.append(nn.Dropout2d(0.8))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConvBlock_Tanh(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn'):
        super(ConvBlock_Tanh, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=2, stride=2, padding=0, output_padding=0, norm='bn'):
        super(DeconvBlock, self).__init__()
        layers = [nn.ConvTranspose2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ResizeBlock(nn.Module):
    def __init__(self, in_fea, out_fea, scale_factor=2, mode='nearest', norm='bn'):
        super(ResizeBlock, self).__init__()
        layers = [nn.Upsample(scale_factor=scale_factor, mode=mode)]
        layers.append(nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=3, stride=1, padding=1))
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)



# FlatFault/CurveFault
# 1000, 70 -> 70, 70
class InversionNet(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0, **kwargs):
        super(InversionNet, self).__init__()
        self.convblock1 = ConvBlock(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = ConvBlock(dim3, dim3, stride=2)
        self.convblock5_2 = ConvBlock(dim3, dim3)
        self.convblock6_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock6_2 = ConvBlock(dim4, dim4)
        self.convblock7_1 = ConvBlock(dim4, dim4, stride=2)
        self.convblock7_2 = ConvBlock(dim4, dim4)
        self.convblock8 = ConvBlock(dim4, dim5, kernel_size=(8, ceil(70 * sample_spatial / 8)), padding=0)
        
        self.deconv1_1 = DeconvBlock(dim5, dim5, kernel_size=5)
        self.deconv1_2 = ConvBlock(dim5, dim5)
        self.deconv2_1 = DeconvBlock(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock(dim4, dim4)
        self.deconv3_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock(dim3, dim3)
        self.deconv4_1 = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock(dim2, dim2)
        self.deconv5_1 = DeconvBlock(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock(dim1, dim1)
        self.deconv6 = ConvBlock_Tanh(dim1, 1)
        
    def forward(self,x):
        # Encoder Part
        x = self.convblock1(x) # (None, 32, 500, 70)
        x = self.convblock2_1(x) # (None, 64, 250, 70)
        x = self.convblock2_2(x) # (None, 64, 250, 70)
        x = self.convblock3_1(x) # (None, 64, 125, 70)
        x = self.convblock3_2(x) # (None, 64, 125, 70)
        x = self.convblock4_1(x) # (None, 128, 63, 70) 
        x = self.convblock4_2(x) # (None, 128, 63, 70)
        x = self.convblock5_1(x) # (None, 128, 32, 35) 
        x = self.convblock5_2(x) # (None, 128, 32, 35)
        x = self.convblock6_1(x) # (None, 256, 16, 18) 
        x = self.convblock6_2(x) # (None, 256, 16, 18)
        x = self.convblock7_1(x) # (None, 256, 8, 9) 
        x = self.convblock7_2(x) # (None, 256, 8, 9)
        x = self.convblock8(x) # (None, 512, 1, 1)
        
        # Decoder Part 
        x = self.deconv1_1(x) # (None, 512, 5, 5)
        x = self.deconv1_2(x) # (None, 512, 5, 5)
        x = self.deconv2_1(x) # (None, 256, 10, 10) 
        x = self.deconv2_2(x) # (None, 256, 10, 10)
        x = self.deconv3_1(x) # (None, 128, 20, 20) 
        x = self.deconv3_2(x) # (None, 128, 20, 20)
        x = self.deconv4_1(x) # (None, 64, 40, 40) 
        x = self.deconv4_2(x) # (None, 64, 40, 40)
        x = self.deconv5_1(x) # (None, 32, 80, 80)
        x = self.deconv5_2(x) # (None, 32, 80, 80)
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0) # (None, 32, 70, 70) 125, 100
        x = self.deconv6(x) # (None, 1, 70, 70)
        return x

class FCN4_Deep_Resize_2(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, ratio=1.0, upsample_mode='nearest'):
        super(FCN4_Deep_Resize_2, self).__init__()
        self.convblock1 = Conv2DwithBN(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = Conv2DwithBN(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = Conv2DwithBN(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = Conv2DwithBN(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = Conv2DwithBN(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = Conv2DwithBN(dim3, dim3, stride=2)
        self.convblock5_2 = Conv2DwithBN(dim3, dim3)
        self.convblock6_1 = Conv2DwithBN(dim3, dim4, stride=2)
        self.convblock6_2 = Conv2DwithBN(dim4, dim4)
        self.convblock7_1 = Conv2DwithBN(dim4, dim4, stride=2)
        self.convblock7_2 = Conv2DwithBN(dim4, dim4)
        self.convblock8 = Conv2DwithBN(dim4, dim5, kernel_size=(8, ceil(70 * ratio / 8)), padding=0)
        
        self.deconv1_1 = ResizeConv2DwithBN(dim5, dim5, scale_factor=5, mode=upsample_mode)
        self.deconv1_2 = Conv2DwithBN(dim5, dim5)
        self.deconv2_1 = ResizeConv2DwithBN(dim5, dim4, scale_factor=2, mode=upsample_mode)
        self.deconv2_2 = Conv2DwithBN(dim4, dim4)
        self.deconv3_1 = ResizeConv2DwithBN(dim4, dim3, scale_factor=2, mode=upsample_mode)
        self.deconv3_2 = Conv2DwithBN(dim3, dim3)
        self.deconv4_1 = ResizeConv2DwithBN(dim3, dim2, scale_factor=2, mode=upsample_mode)
        self.deconv4_2 = Conv2DwithBN(dim2, dim2)
        self.deconv5_1 = ResizeConv2DwithBN(dim2, dim1, scale_factor=2, mode=upsample_mode)
        self.deconv5_2 = Conv2DwithBN(dim1, dim1)
        self.deconv6 = Conv2DwithBN_Tanh(dim1, 1)
        
    def forward(self,x):
        # Encoder Part
        x = self.convblock1(x) # (None, 32, 500, 70)
        x = self.convblock2_1(x) # (None, 64, 250, 70)
        x = self.convblock2_2(x) # (None, 64, 250, 70)
        x = self.convblock3_1(x) # (None, 64, 125, 70)
        x = self.convblock3_2(x) # (None, 64, 125, 70)
        x = self.convblock4_1(x) # (None, 128, 63, 70)
        x = self.convblock4_2(x) # (None, 128, 63, 70)
        x = self.convblock5_1(x) # (None, 128, 32, 35)
        x = self.convblock5_2(x) # (None, 128, 32, 35)
        x = self.convblock6_1(x) # (None, 256, 16, 18)
        x = self.convblock6_2(x) # (None, 256, 16, 18)
        x = self.convblock7_1(x) # (None, 256, 8, 9)
        x = self.convblock7_2(x) # (None, 256, 8, 9)
        x = self.convblock8(x) # (None, 512, 1, 1)
        
        # Decoder Part 
        x = self.deconv1_1(x) # (None, 512, 5, 5)
        x = self.deconv1_2(x) # (None, 512, 5, 5)
        x = self.deconv2_1(x) # (None, 256, 10, 10)
        x = self.deconv2_2(x) # (None, 256, 10, 10)
        x = self.deconv3_1(x) # (None, 128, 20, 20)
        x = self.deconv3_2(x) # (None, 128, 20, 20)
        x = self.deconv4_1(x) # (None, 64, 40, 40)
        x = self.deconv4_2(x) # (None, 64, 40, 40)
        x = self.deconv5_1(x) # (None, 32, 80, 80)
        x = self.deconv5_2(x) # (None, 32, 80, 80)
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0) # (None, 32, 70, 70)
        x = self.deconv6(x) # (None, 1, 70, 70)
        return x


class Discriminator(nn.Module):
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, **kwargs):
        super(Discriminator, self).__init__()
        self.convblock1_1 = ConvBlock(1, dim1, stride=2)
        self.convblock1_2 = ConvBlock(dim1, dim1)
        self.convblock2_1 = ConvBlock(dim1, dim2, stride=2)
        self.convblock2_2 = ConvBlock(dim2, dim2)
        self.convblock3_1 = ConvBlock(dim2, dim3, stride=2)
        self.convblock3_2 = ConvBlock(dim3, dim3)
        self.convblock4_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock4_2 = ConvBlock(dim4, dim4)
        self.convblock5 = ConvBlock(dim4, 1, kernel_size=5, padding=0)

    def forward(self, x):
        x = self.convblock1_1(x)
        x = self.convblock1_2(x)
        x = self.convblock2_1(x)
        x = self.convblock2_2(x)
        x = self.convblock3_1(x)
        x = self.convblock3_2(x)
        x = self.convblock4_1(x)
        x = self.convblock4_2(x)
        x = self.convblock5(x)
        x = x.view(x.shape[0], -1)
        return x


class Conv_HPGNN(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=None, stride=None, padding=None, **kwargs):
        super(Conv_HPGNN, self).__init__()
        layers = [
            ConvBlock(in_fea, out_fea, relu_slop=0.1, dropout=0.8),
            ConvBlock(out_fea, out_fea, relu_slop=0.1, dropout=0.8),
        ]
        if kernel_size is not None:
            layers.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Deconv_HPGNN(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size, **kwargs):
        super(Deconv_HPGNN, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_fea, in_fea, kernel_size=kernel_size, stride=2, padding=0),
            ConvBlock(in_fea, out_fea, relu_slop=0.1, dropout=0.8),
            ConvBlock(out_fea, out_fea, relu_slop=0.1, dropout=0.8)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    """Transformer encoder block with pre-norm for improved stability."""

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.drop(attn_out)
        x = x + self.mlp(self.norm2(x))
        return x


class SeismicViT(nn.Module):
    """Vision Transformer based architecture for seismic FWI.

    Input:  (bs, 5, 1000, 70) – seismic data
    Output: (bs, 1, 70,   70) – velocity model

    Architecture:
    1. CNN stem – progressive temporal then joint spatial+temporal compression
       with spectral normalisation and GroupNorm for training stability.
    2. Flatten feature map to token sequence.
    3. Learnable positional embeddings + Transformer encoder.
    4. Bilinear-upsample decoder with skip connections from the encoder.
    """

    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, embed_dim=512,
                 num_heads=8, num_layers=6, mlp_ratio=4.0, dropout=0.1,
                 sample_spatial=1.0, **kwargs):
        super().__init__()

        # ── CNN Encoder ──────────────────────────────────────────────────
        # Stages 1-4: temporal-only compression ((k,1) kernels, stride (2,1))
        self.enc1 = self._conv_block(5,    dim1, (7, 1), (2, 1), (3, 0))  # (bs,  32, 500, 70)
        self.enc2 = self._conv_block(dim1, dim2, (3, 1), (2, 1), (1, 0))  # (bs,  64, 250, 70)
        self.enc3 = self._conv_block(dim2, dim2, (3, 1), (2, 1), (1, 0))  # (bs,  64, 125, 70)
        self.enc4 = self._conv_block(dim2, dim3, (3, 1), (2, 1), (1, 0))  # (bs, 128,  63, 70)
        # Stages 5-7: joint spatial+temporal compression (3×3 kernels, stride 2)
        self.enc5 = self._conv_block(dim3, dim3, 3, 2, 1)                  # (bs, 128,  32, 35)
        self.enc6 = self._conv_block(dim3, dim4, 3, 2, 1)                  # (bs, 256,  16, 18)
        self.enc7 = self._conv_block(dim4, embed_dim, 3, 2, 1)             # (bs, 512,   8,  9)

        self._h_tokens = 8
        self._w_tokens = 9
        num_tokens = self._h_tokens * self._w_tokens  # 72

        # ── Positional embeddings ─────────────────────────────────────────
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ── Transformer encoder ───────────────────────────────────────────
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.transformer_norm = nn.LayerNorm(embed_dim)

        # ── Decoder with skip connections ─────────────────────────────────
        # dec1: (512, 8, 9) → upsample (16,18) → cat skip from enc6 → (256, 16, 18)
        self.dec1_proj = nn.Conv2d(embed_dim, dim4, 1)
        self.dec1_up   = nn.Upsample(size=(16, 18), mode='bilinear', align_corners=False)
        self.dec1_conv = self._dec_block(dim4 + dim4, dim4)

        # dec2: (256,16,18) → upsample (35,35) → cat skip from enc5 → (128, 35, 35)
        self.dec2_proj = nn.Conv2d(dim4, dim3, 1)
        self.dec2_up   = nn.Upsample(size=(35, 35), mode='bilinear', align_corners=False)
        self.dec2_conv = self._dec_block(dim3 + dim3, dim3)

        # dec3: (128,35,35) → upsample (70,70) → (64, 70, 70)
        self.dec3_proj = nn.Conv2d(dim3, dim2, 1)
        self.dec3_up   = nn.Upsample(size=(70, 70), mode='bilinear', align_corners=False)
        self.dec3_conv = self._dec_block(dim2, dim2)

        # Final: (64,70,70) → (1,70,70)
        self.out_conv = nn.Sequential(
            nn.Conv2d(dim2, dim2 // 2, 3, padding=1),
            nn.GroupNorm(self._num_groups(dim2 // 2), dim2 // 2),
            nn.GELU(),
            nn.Conv2d(dim2 // 2, 1, 1),
            nn.Tanh(),
        )

        self._init_weights()

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _num_groups(channels, max_groups=8):
        """Return a valid num_groups for GroupNorm(channels)."""
        g = max_groups
        while g > 1 and channels % g != 0:
            g //= 2
        return g

    def _conv_block(self, in_ch, out_ch, kernel, stride, pad):
        """Conv2d (spectral-normed) + GroupNorm + GELU."""
        conv = nn.Conv2d(in_ch, out_ch, kernel, stride, pad)
        nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
        if conv.bias is not None:
            nn.init.zeros_(conv.bias)
        return nn.Sequential(
            nn.utils.spectral_norm(conv),
            nn.GroupNorm(self._num_groups(out_ch), out_ch),
            nn.GELU(),
        )

    def _dec_block(self, in_ch, out_ch):
        """Double-conv decoder block with GroupNorm + GELU."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(self._num_groups(out_ch), out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(self._num_groups(out_ch), out_ch),
            nn.GELU(),
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, x):
        # Encode (save skips from enc5 and enc6)
        x = self.enc1(x)   # (bs,  32, 500, 70)
        x = self.enc2(x)   # (bs,  64, 250, 70)
        x = self.enc3(x)   # (bs,  64, 125, 70)
        x = self.enc4(x)   # (bs, 128,  63, 70)
        s5 = self.enc5(x)  # (bs, 128,  32, 35)  ← skip 1
        s6 = self.enc6(s5) # (bs, 256,  16, 18)  ← skip 2
        z  = self.enc7(s6) # (bs, 512,   8,  9)

        # Tokenise: (bs, 512, 8, 9) → (bs, 72, 512)
        bs, C, H, W = z.shape
        tokens = z.flatten(2).permute(0, 2, 1) + self.pos_embed

        # Transformer
        for block in self.transformer_blocks:
            tokens = block(tokens)
        tokens = self.transformer_norm(tokens)

        # Reshape back: (bs, 72, 512) → (bs, 512, 8, 9)
        z = tokens.permute(0, 2, 1).reshape(bs, C, H, W)

        # Decode with skip connections
        d1 = self.dec1_up(self.dec1_proj(z))              # (bs, 256, 16, 18)
        d1 = torch.cat([d1, s6], dim=1)                   # (bs, 512, 16, 18)
        d1 = self.dec1_conv(d1)                            # (bs, 256, 16, 18)

        d2 = self.dec2_up(self.dec2_proj(d1))             # (bs, 128, 35, 35)
        s5_resized = F.interpolate(s5, size=(35, 35), mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, s5_resized], dim=1)                  # (bs, 256, 35, 35)
        d2 = self.dec2_conv(d2)                            # (bs, 128, 35, 35)

        d3 = self.dec3_up(self.dec3_proj(d2))             # (bs,  64, 70, 70)
        d3 = self.dec3_conv(d3)                            # (bs,  64, 70, 70)

        return self.out_conv(d3)                           # (bs,   1, 70, 70)


model_dict = {
    'InversionNet': InversionNet,
    'Discriminator': Discriminator,
    'UPFWI': FCN4_Deep_Resize_2,
    'SeismicViT': SeismicViT,
}

