"""
## Towards Real-World Burst Image Super-Resolution: Benchmark and Method
## Code by YujingSun
"""
import cv2
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import numpy as np
import time
from torch import einsum
import common

from torchvision import transforms
import seaborn as sns
import matplotlib.pyplot as plt
import os


#####################################################################################
################################### Basic Layers ####################################
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x
        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, strides=1):
        super(ConvBlock, self).__init__()
        self.strides = strides
        self.in_channel=in_channel
        self.out_channel=out_channel
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.conv11 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, padding=0)

    def forward(self, x):
        out1 = self.block(x)
        out2 = self.conv11(x)
        out = out1 + out2
        return out

    def flops(self, H, W):
        flops = H*W*self.in_channel*self.out_channel*(3*3+1)+H*W*self.out_channel*self.out_channel*3*3
        return flops

class UNet(nn.Module):
    def __init__(self, block=ConvBlock,dim=32):
        super(UNet, self).__init__()

        self.dim = dim
        self.ConvBlock1 = ConvBlock(3, dim, strides=1)
        self.pool1 = nn.Conv2d(dim,dim,kernel_size=4, stride=2, padding=1)

        self.ConvBlock2 = block(dim, dim*2, strides=1)
        self.pool2 = nn.Conv2d(dim*2,dim*2,kernel_size=4, stride=2, padding=1)

        self.ConvBlock3 = block(dim*2, dim*4, strides=1)
        self.pool3 = nn.Conv2d(dim*4,dim*4,kernel_size=4, stride=2, padding=1)

        self.ConvBlock4 = block(dim*4, dim*8, strides=1)
        self.pool4 = nn.Conv2d(dim*8, dim*8,kernel_size=4, stride=2, padding=1)

        self.ConvBlock5 = block(dim*8, dim*16, strides=1)

        self.upv6 = nn.ConvTranspose2d(dim*16, dim*8, 2, stride=2)
        self.ConvBlock6 = block(dim*16, dim*8, strides=1)

        self.upv7 = nn.ConvTranspose2d(dim*8, dim*4, 2, stride=2)
        self.ConvBlock7 = block(dim*8, dim*4, strides=1)

        self.upv8 = nn.ConvTranspose2d(dim*4, dim*2, 2, stride=2)
        self.ConvBlock8 = block(dim*4, dim*2, strides=1)

        self.upv9 = nn.ConvTranspose2d(dim*2, dim, 2, stride=2)
        self.ConvBlock9 = block(dim*2, dim, strides=1)

        self.conv10 = nn.Conv2d(dim, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        conv1 = self.ConvBlock1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.ConvBlock2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.ConvBlock3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.ConvBlock4(pool3)
        pool4 = self.pool4(conv4)

        conv5 = self.ConvBlock5(pool4)

        up6 = self.upv6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.ConvBlock6(up6)

        up7 = self.upv7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.ConvBlock7(up7)

        up8 = self.upv8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.ConvBlock8(up8)

        up9 = self.upv9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.ConvBlock9(up9)

        conv10 = self.conv10(conv9)
        out = x + conv10

        return out

    def flops(self, H, W):
        flops = 0
        flops += self.ConvBlock1.flops(H, W)
        flops += H/2*W/2*self.dim*self.dim*4*4
        flops += self.ConvBlock2.flops(H/2, W/2)
        flops += H/4*W/4*self.dim*2*self.dim*2*4*4
        flops += self.ConvBlock3.flops(H/4, W/4)
        flops += H/8*W/8*self.dim*4*self.dim*4*4*4
        flops += self.ConvBlock4.flops(H/8, W/8)
        flops += H/16*W/16*self.dim*8*self.dim*8*4*4

        flops += self.ConvBlock5.flops(H/16, W/16)

        flops += H/8*W/8*self.dim*16*self.dim*8*2*2
        flops += self.ConvBlock6.flops(H/8, W/8)
        flops += H/4*W/4*self.dim*8*self.dim*4*2*2
        flops += self.ConvBlock7.flops(H/4, W/4)
        flops += H/2*W/2*self.dim*4*self.dim*2*2*2
        flops += self.ConvBlock8.flops(H/2, W/2)
        flops += H*W*self.dim*2*self.dim*2*2
        flops += self.ConvBlock9.flops(H, W)

        flops += H*W*self.dim*3*3*3
        return flops

######################################################################################
######################################################################################


######################################################################################
############################# Basic Layers of Transformer ############################

class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim))
        self.s = s

    def forward(self, x, H=None, W=None):
        B, N, C = x.shape
        H = H or int(math.sqrt(N))
        W = W or int(math.sqrt(N))
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]

class SELayer_ori(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer_ori, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: [B, N, C]
        x = torch.transpose(x, 1, 2)  # [B, C, N]
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        x = x * y.expand_as(x)
        x = torch.transpose(x, 1, 2)  # [B, N, C]
        return x

class SepConv2d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,act_layer=nn.ReLU):
        super(SepConv2d, self).__init__()
        self.depthwise = torch.nn.Conv2d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=in_channels)
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act_layer = act_layer() if act_layer is not None else nn.Identity()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise(x)
        return x

    def flops(self, H, W):
        flops = 0
        flops += H*W*self.in_channels*self.kernel_size**2/self.stride**2
        flops += H*W*self.in_channels*self.out_channels
        return flops

class ConvProjection(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout = 0.,
                 last_stage=False,bias=True):

        super().__init__()

        inner_dim = dim_head *  heads
        self.heads = heads
        pad = (kernel_size - q_stride)//2
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad, bias)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad, bias)
        self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad, bias)

    def forward(self, x, attn_kv=None):
        b, n, c, h = *x.shape, self.heads
        l = int(math.sqrt(n))
        w = int(math.sqrt(n))

        attn_kv = x if attn_kv is None else attn_kv
        x = rearrange(x, 'b (l w) c -> b c l w', l=l, w=w)
        attn_kv = rearrange(attn_kv, 'b (l w) c -> b c l w', l=l, w=w)
        # print(attn_kv)
        q = self.to_q(x)
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)

        k = self.to_k(attn_kv)
        v = self.to_v(attn_kv)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)
        return q,k,v

    def flops(self, H, W):
        flops = 0
        flops += self.to_q.flops(H, W)
        flops += self.to_k.flops(H, W)
        flops += self.to_v.flops(H, W)
        return flops

class LinearProjection(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., bias=True):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias = bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        attn_kv = x if attn_kv is None else attn_kv
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1]
        return q,k,v

    def flops(self, H, W):
        flops = H*W*self.dim*self.inner_dim*3
        return flops

class LinearProjection_Concat_kv(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., bias=True):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        attn_kv = x if attn_kv is None else attn_kv
        qkv_dec = self.to_qkv(x).reshape(B_, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv_enc = self.to_kv(attn_kv).reshape(B_, N, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k_d, v_d = qkv_dec[0], qkv_dec[1], qkv_dec[2]  # make torchscript happy (cannot use tensor as tuple)
        k_e, v_e = kv_enc[0], kv_enc[1]
        k = torch.cat((k_d,k_e),dim=2)
        v = torch.cat((v_d,v_e),dim=2)
        return q,k,v

    def flops(self, H, W):
        flops = H*W*self.dim*self.inner_dim*5
        return flops

class WindowAttention(nn.Module):
    def __init__(self, dim, win_size,num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,se_layer=False):

        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0]) # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1]) # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if token_projection =='conv':
            self.qkv = ConvProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
        elif token_projection =='linear_concat':
            self.qkv = LinearProjection_Concat_kv(dim,num_heads,dim//num_heads,bias=qkv_bias)
        else:
            self.qkv = LinearProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)

        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.se_layer = SELayer(dim) if se_layer else nn.Identity()
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x,attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ratio = attn.size(-1)//relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d = ratio)

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)',d = ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N*ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N*ratio)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.se_layer(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}'

    def flops(self, H, W):
        # calculate flops for 1 window with token length of N
        # print(N, self.dim)
        flops = 0
        N = self.win_size[0]*self.win_size[1]
        nW = H*W/N
        # qkv = self.qkv(x)
        # flops += N * self.dim * 3 * self.dim
        flops += self.qkv.flops(H, W)
        # attn = (q @ k.transpose(-2, -1))
        if self.token_projection !='linear_concat':
            flops += nW * self.num_heads * N * (self.dim // self.num_heads) * N
            #  x = (attn @ v)
            flops += nW * self.num_heads * N * N * (self.dim // self.num_heads)
        else:
            flops += nW * self.num_heads * N * (self.dim // self.num_heads) * N*2
            #  x = (attn @ v)
            flops += nW * self.num_heads * N * N*2 * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += nW * N * self.dim * self.dim
        print("W-MSA:{%.2f}"%(flops/1e9))
        return flops

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def flops(self, H, W):
        flops = 0
        # fc1
        flops += H*W*self.in_features*self.hidden_features
        # fc2
        flops += H*W*self.hidden_features*self.out_features
        print("MLP:{%.2f}"%(flops/1e9))
        return flops

class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU,drop = 0.):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                act_layer())
        self.dwconv = nn.Sequential(nn.Conv2d(hidden_dim,hidden_dim,groups=hidden_dim,kernel_size=3,stride=1,padding=1),
                        act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # bs x hw x c
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h = hh, w = hh)
        # bs,hidden_dim,32x32

        x = self.dwconv(x)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h = hh, w = hh)

        x = self.linear2(x)

        return x

    def flops(self, H, W):
        flops = 0
        # fc1
        flops += H*W*self.dim*self.hidden_dim
        # dwconv
        flops += H*W*self.hidden_dim*3*3
        # fc2
        flops += H*W*self.hidden_dim*self.dim
        print("LeFF:{%.2f}"%(flops/1e9))
        return flops

def window_partition(x, win_size, dilation_rate=1):
    B, H, W, C = x.shape
    if dilation_rate !=1:
        x = x.permute(0,3,1,2) # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size,dilation=dilation_rate,padding=4*(dilation_rate-1),stride=win_size) # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0,2,1).contiguous().view(-1, C, win_size, win_size) # B' ,C ,Wh ,Ww
        windows = windows.permute(0,2,3,1).contiguous() # B' ,Wh ,Ww ,C
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C) # B' ,Wh ,Ww ,C
    return windows

def window_reverse(windows, win_size, H, W, dilation_rate=1):
    # B' ,Wh ,Ww ,C
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate !=1:
        x = windows.permute(0,5,3,4,1,2).contiguous() # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4*(dilation_rate-1),stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        # import pdb;pdb.set_trace()
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.conv(x).flatten(2).transpose(1,2).contiguous()  # B H*W C
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H/2*W/2*self.in_channel*self.out_channel*4*4
        print("Downsample:{%.2f}"%(flops/1e9))
        return flops

class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x).flatten(2).transpose(1,2).contiguous() # B H*W C
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*2*W*2*self.in_channel*self.out_channel*2*2
        print("Upsample:{%.2f}"%(flops/1e9))
        return flops

class Downsample_flatten(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample_flatten, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, C, H, W = x.shape
        # import pdb;pdb.set_trace()
        out = self.conv(x).contiguous()  # B H*W C
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H/2*W/2*self.in_channel*self.out_channel*4*4
        print("Downsample:{%.2f}"%(flops/1e9))
        return flops

class Upsample_flatten(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample_flatten, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, C, H, W = x.shape
        out = self.deconv(x).contiguous() # B H*W C
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*2*W*2*self.in_channel*self.out_channel*2*2
        print("Upsample:{%.2f}"%(flops/1e9))
        return flops

class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1, norm_layer=None,act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
            act_layer(inplace=True)
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*W*self.in_channel*self.out_channel*3*3

        if self.norm is not None:
            flops += H*W*self.out_channel
        print("Input_proj:{%.2f}"%(flops/1e9))
        return flops

class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1, norm_layer=None,act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*W*self.in_channel*self.out_channel*3*3

        if self.norm is not None:
            flops += H*W*self.out_channel
        print("Output_proj:{%.2f}"%(flops/1e9))
        return flops

class OutputProj_HWC(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1, norm_layer=None,act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = self.proj(x.transpose(1, 2).view(B, C, H, W))
        if self.norm is not None:
            x = self.norm(x)
        x = x.view(B, -1, H*W).transpose(1, 2).view(B, H*W, -1)
        return x

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*W*self.in_channel*self.out_channel*3*3

        if self.norm is not None:
            flops += H*W*self.out_channel
        print("Output_proj:{%.2f}"%(flops/1e9))
        return flops

######################################################################################
######################################################################################


######################################################################################
################################## FAF Block #########################################
class NewFusion(nn.Module):
    def __init__(self, num_feat=64, num_frame=14, center_frame_idx=0):
        super(NewFusion, self).__init__()

        '''
        # Compuate the attention map, highlight distinctions while keep similarities
        
        Input: Aligned frames, [B, T, C, H, W]
        Output: Fused frame, [B, C, H, W]
        '''

        self.center_frame_idx = center_frame_idx
        self.temporal_attn1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.temporal_attn2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.feat_fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        # spatial attention
        self.downsample1 = Downsample_flatten(num_feat, num_feat*2)
        self.downsample2 = Downsample_flatten(num_feat*2, num_feat*4)

        self.upsample1 = Upsample_flatten(num_feat*4, num_feat*2)
        self.upsample2 = Upsample_flatten(num_feat*4, num_feat)

        n_resblocks = 2
        conv = common.default_conv
        m_res_block1 = [
            common.ResBlock(
                conv, num_feat, kernel_size=3
            ) for _ in range(n_resblocks)
        ]

        m_res_block2 = [
            common.ResBlock(
                conv, num_feat*2, kernel_size=3
            ) for _ in range(n_resblocks)
        ]

        m_res_block3 = [
            common.ResBlock(
                conv, num_feat*4, kernel_size=3
            ) for _ in range(n_resblocks)
        ]

        m_res_block4 = [
            common.ResBlock(
                conv, num_feat*4, kernel_size=3
            ) for _ in range(n_resblocks)
        ]

        m_res_block5 = [
            common.ResBlock(
                conv, num_feat*2, kernel_size=3
            ) for _ in range(n_resblocks)
        ]

        m_fusion_tail = [conv(num_feat*2, num_feat, kernel_size=3)]

        self.res_block1 = nn.Sequential(*m_res_block1)
        self.res_block2 = nn.Sequential(*m_res_block2)
        self.res_block3 = nn.Sequential(*m_res_block3)
        self.res_block4 = nn.Sequential(*m_res_block4)
        self.res_block5 = nn.Sequential(*m_res_block5)
        self.fusion_tail = nn.Sequential(*m_fusion_tail)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def forward(self, aligned_feat):
        b, t, c, h, w = aligned_feat.size()

        # attention map, highlight distinctions while keep similarities
        embedding_ref = self.temporal_attn1(aligned_feat[:, self.center_frame_idx, :, :, :].clone())
        embedding = self.temporal_attn2(aligned_feat.view(-1, c, h, w))
        embedding = embedding.view(b, t, -1, h, w)  # [b,t,c,h,w]

        corr_diff = []
        corr_l = []
        for i in range(t):
            emb_neighbor = embedding[:, i, :, :, :]
            corr = torch.sum(emb_neighbor * embedding_ref, 1).unsqueeze(1)  # [b,1,h,w]
            corr_l.append(corr)
            if i == 0:
                continue
            else:
                # compute the difference among each frame and the base frame
                corr_difference = torch.abs(corr_l[i] - corr_l[0])
                corr_diff.append(corr_difference)
        corr_l_cat = torch.cat(corr_l, dim=1)

        # compute the attention map
        corr_prob = torch.sigmoid(torch.cat(corr_diff, dim=1))  # [b,t-1,h,w]
        
        corr_prob = corr_prob.unsqueeze(2).expand(b, t-1, c, h, w)  # [b,t,c,h,w]
        corr_prob = corr_prob.contiguous().view(b, -1, h, w)  # [b,(t-1)*c,h,w]
        
        aligned_oth_feat = aligned_feat[:, 1 : 14, :, :, :]
        aligned_oth_feat = aligned_oth_feat.view(b, -1, h, w) * corr_prob
        
        aligned_feat_guided = torch.zeros(b, t*c, h, w).to('cuda')
        aligned_feat_guided[:, 0 : c, :, :] = aligned_feat[:, 0 : 1, :, :, :].view(b, -1, h, w)
        aligned_feat_guided[:, c : t*c, :, :] = aligned_oth_feat
        
        #fuse the feat under the guidance of computed attention map
        feat = self.lrelu(self.feat_fusion(aligned_feat_guided))  # [b,c,h,w]

        # Hourglass for spatial attention
        feat_res1 = self.res_block1(feat)
        down_feat1 = self.downsample1(feat_res1)
        feat_res2 = self.res_block2(down_feat1)
        down_feat2 = self.downsample2(feat_res2)

        feat3 = self.res_block3(down_feat2)

        up_feat3 = self.upsample1(feat3)
        concat_2_1 = torch.cat([up_feat3, feat_res2], 1)
        feat_res4 = self.res_block4(concat_2_1)
        up_feat4 = self.upsample2(feat_res4)
        concat_1_0 = torch.cat([up_feat4, feat_res1], 1)
        feat_res5 = self.res_block5(concat_1_0)

        feat_out = self.fusion_tail(feat_res5) + feat

        return feat_out

######################################################################################
################################## BasicTransformerBlock #############################
class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, win_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,token_projection='linear',token_mlp='leff',se_layer=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.token_mlp = token_mlp
        if min(self.input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-win_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, win_size=to_2tuple(self.win_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            token_projection=token_projection,se_layer=se_layer)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,act_layer=act_layer, drop=drop) if token_mlp=='ffn' else LeFF(dim,mlp_hidden_dim,act_layer=act_layer, drop=drop)


    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"win_size={self.win_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def forward(self, x, mask=None):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))


        ## input mask
        if mask != None:
            input_mask = F.interpolate(mask, size=(H,W)).permute(0,2,3,1)
            input_mask_windows = window_partition(input_mask, self.win_size) # nW, win_size, win_size, 1
            attn_mask = input_mask_windows.view(-1, self.win_size * self.win_size) # nW, win_size*win_size
            attn_mask = attn_mask.unsqueeze(2)*attn_mask.unsqueeze(1) # nW, win_size*win_size, win_size*win_size
            attn_mask = attn_mask.masked_fill(attn_mask!=0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        ## shift mask
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            shift_mask = torch.zeros((1, H, W, 1)).type_as(x)
            h_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1
            shift_mask_windows = window_partition(shift_mask, self.win_size)  # nW, win_size, win_size, 1
            shift_mask_windows = shift_mask_windows.view(-1, self.win_size * self.win_size) # nW, win_size*win_size
            shift_attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(2) # nW, win_size*win_size, win_size*win_size
            shift_attn_mask = shift_attn_mask.masked_fill(shift_attn_mask != 0, float(-100.0)).masked_fill(shift_attn_mask == 0, float(0.0))
            attn_mask = attn_mask + shift_attn_mask if attn_mask is not None else shift_attn_mask

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        x_windows = x_windows.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, win_size*win_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
        shifted_x = window_reverse(attn_windows, self.win_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        del attn_mask
        return x

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        flops += self.attn.flops(H, W)
        # norm2
        flops += self.dim * H * W
        # mlp
        flops += self.mlp.flops(H,W)
        print("LeWin:{%.2f}"%(flops/1e9))
        return flops


#######################################################################################
################################## Basiclayer of BaseModel ###########################
class BasicBaseModelLayer(nn.Module):
    def __init__(self, dim, output_dim, input_resolution, depth, num_heads, win_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, use_checkpoint=False,
                 token_projection='linear',token_mlp='ffn',se_layer=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # build blocks
        self.blocks = nn.ModuleList([
            BasicTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, win_size=win_size,
                                 shift_size=0 if (i % 2 == 0) else win_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
            for i in range(depth)])

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def forward(self, x, mask=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x,mask)
        return x

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops


#######################################################################################
################################## BaseModel (FBANet) ###########################
class BaseModel(nn.Module):
    def __init__(self, patch_size=0.0, img_size=128, in_chans=3,
                 embed_dim=32, depths=[2, 2, 2, 2, 2, 2, 2, 2, 2], num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
                 win_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, token_projection='linear', token_mlp='ffn', se_layer=False,
                 dowsample=Downsample, upsample=Upsample, **kwargs):
        super().__init__()

        self.num_enc_layers = len(depths)//2
        self.num_dec_layers = len(depths)//2
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.win_size =win_size
        self.reso = img_size
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.num_frames = 14

        # stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))]
        conv_dpr = [drop_path_rate]*depths[4]
        dec_dpr = enc_dpr[::-1]

        # build layers

        # Input/Output
        self.input_proj = InputProj(in_channel=embed_dim, out_channel=embed_dim, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
        self.output_proj = OutputProj_HWC(in_channel=2*embed_dim, out_channel=embed_dim, kernel_size=3, stride=1)
        self.output_proj_2 = OutputProj(in_channel=2*embed_dim, out_channel=embed_dim, kernel_size=3, stride=1)

        self.output_proj_HG1_0 = OutputProj_HWC(in_channel=2*embed_dim, out_channel=embed_dim, kernel_size=3, stride=1)

        self.output_proj_HG2_0 = OutputProj_HWC(in_channel=8*embed_dim, out_channel=4*embed_dim, kernel_size=3, stride=1)
        self.output_proj_HG2_1 = OutputProj_HWC(in_channel=4*embed_dim, out_channel=2*embed_dim, kernel_size=3, stride=1)


        self.se_block_HG1_0 = SEBasicBlock(inplanes=embed_dim*4, planes=embed_dim*4)
        self.se_block_HG1_1 = SEBasicBlock(inplanes=embed_dim*2, planes=embed_dim*2)

        self.se_block_HG2_0 = SEBasicBlock(inplanes=embed_dim*4, planes=embed_dim*4)
        self.se_block_HG2_1 = SEBasicBlock(inplanes=embed_dim*2, planes=embed_dim*2)

        # # pixel_mask
        # self.pixel_mask = PixelMask(height=img_size, width=img_size, dim=embed_dim)

        conv = common.default_conv
        scale = 4
        n_resblocks = 2

        m_head = [conv(in_chans, embed_dim, kernel_size=3)]

        m_body = [
            common.ResBlock(
                conv, embed_dim, kernel_size=3
            ) for _ in range(n_resblocks)
        ]

        self.fusion = NewFusion(num_feat=embed_dim, num_frame=self.num_frames, center_frame_idx=0)

        m_tail = [
            common.Upsampler(conv, scale, embed_dim, act=False),
            conv(embed_dim, in_chans, kernel_size=3)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)


        ###### HG Block1 ######
        # Encoder
        HG1_res_enc_0 = [
            common.ResBlock(
                conv, embed_dim, kernel_size=3
            ) for _ in range(n_resblocks)
        ]
        self.HG1_res_enc0 = nn.Sequential(*HG1_res_enc_0)
        self.HG1_encoderlayer_0 = BasicBaseModelLayer(dim=embed_dim,
                                                    output_dim=embed_dim,
                                                    input_resolution=(img_size,
                                                                      img_size),
                                                    depth=depths[0],
                                                    num_heads=num_heads[0],
                                                    win_size=win_size,
                                                    mlp_ratio=self.mlp_ratio,
                                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                                    drop_path=enc_dpr[sum(depths[:0]):sum(depths[:1])],
                                                    norm_layer=norm_layer,
                                                    use_checkpoint=use_checkpoint,
                                                    token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        self.HG1_conv_enc0 = nn.Sequential(conv(embed_dim, embed_dim, kernel_size=3))

        self.HG1_dowsample_0 = dowsample(embed_dim, embed_dim*2)
        HG1_res_enc_1 = [
            common.ResBlock(
                conv, embed_dim*2, kernel_size=3
            ) for _ in range(n_resblocks)
        ]
        self.HG1_res_enc1 = nn.Sequential(*HG1_res_enc_1)
        self.HG1_encoderlayer_1 = BasicBaseModelLayer(dim=embed_dim*2,
                                                    output_dim=embed_dim*2,
                                                    input_resolution=(img_size // 2,
                                                                      img_size // 2),
                                                    depth=depths[1],
                                                    num_heads=num_heads[1],
                                                    win_size=win_size,
                                                    mlp_ratio=self.mlp_ratio,
                                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                                    drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                                                    norm_layer=norm_layer,
                                                    use_checkpoint=use_checkpoint,
                                                    token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        self.HG1_conv_enc1 = nn.Sequential(conv(embed_dim*2, embed_dim*2, kernel_size=3))

        self.HG1_dowsample_1 = dowsample(embed_dim*2, embed_dim*4)

        # Bottleneck
        self.conv_HG1 = BasicBaseModelLayer(dim=embed_dim*4,
                                          output_dim=embed_dim*4,
                                          input_resolution=(img_size // (2 ** 2),
                                                            img_size // (2 ** 2)),
                                          depth=depths[4],
                                          num_heads=num_heads[4],
                                          win_size=win_size,
                                          mlp_ratio=self.mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          drop=drop_rate, attn_drop=attn_drop_rate,
                                          drop_path=conv_dpr,
                                          norm_layer=norm_layer,
                                          use_checkpoint=use_checkpoint,
                                          token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)

        # Decoder
        self.HG1_upsample_0 = upsample(embed_dim*4, embed_dim*2)
        HG1_res_dec_0 = [
            common.ResBlock(
                conv, embed_dim*4, kernel_size=3
            ) for _ in range(n_resblocks)
        ]
        self.HG1_res_dec0 = nn.Sequential(*HG1_res_dec_0)
        self.HG1_decoderlayer_0 = BasicBaseModelLayer(dim=embed_dim*4,
                                                    output_dim=embed_dim*4,
                                                    input_resolution=(img_size // 2,
                                                                      img_size // 2),
                                                    depth=depths[5],
                                                    num_heads=num_heads[5],
                                                    win_size=win_size,
                                                    mlp_ratio=self.mlp_ratio,
                                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                                    drop_path=dec_dpr[:depths[5]],
                                                    norm_layer=norm_layer,
                                                    use_checkpoint=use_checkpoint,
                                                    token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)

        self.HG1_conv_dec0 = nn.Sequential(conv(embed_dim*4, embed_dim*4, kernel_size=3))

        self.HG1_upsample_1 = upsample(embed_dim*4, embed_dim)
        HG1_res_dec_1 = [
            common.ResBlock(
                conv, embed_dim * 2, kernel_size=3
            ) for _ in range(n_resblocks)
        ]
        self.HG1_res_dec1 = nn.Sequential(*HG1_res_dec_1)
        self.HG1_decoderlayer_1 = BasicBaseModelLayer(dim=embed_dim*2,
                                                    output_dim=embed_dim*2,
                                                    input_resolution=(img_size,
                                                                      img_size),
                                                    depth=depths[6],
                                                    num_heads=num_heads[6],
                                                    win_size=win_size,
                                                    mlp_ratio=self.mlp_ratio,
                                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                                    drop_path=dec_dpr[sum(depths[5:6]):sum(depths[5:7])],
                                                    norm_layer=norm_layer,
                                                    use_checkpoint=use_checkpoint,
                                                    token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        self.HG1_conv_dec1 = nn.Sequential(conv(embed_dim * 2, embed_dim * 2, kernel_size=3))

        ###### HG Block2 ######
        # Encoder
        HG2_res_enc_0 = [
            common.ResBlock(
                conv, embed_dim, kernel_size=3
            ) for _ in range(n_resblocks)
        ]
        self.HG2_res_enc0 = nn.Sequential(*HG2_res_enc_0)
        self.HG2_encoderlayer_0 = BasicBaseModelLayer(dim=embed_dim,
                                                    output_dim=embed_dim,
                                                    input_resolution=(img_size,
                                                                      img_size),
                                                    depth=depths[0],
                                                    num_heads=num_heads[0],
                                                    win_size=win_size,
                                                    mlp_ratio=self.mlp_ratio,
                                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                                    drop_path=enc_dpr[sum(depths[:0]):sum(depths[:1])],
                                                    norm_layer=norm_layer,
                                                    use_checkpoint=use_checkpoint,
                                                    token_projection=token_projection, token_mlp=token_mlp,
                                                    se_layer=se_layer)
        self.HG2_conv_enc0 = nn.Sequential(conv(embed_dim, embed_dim, kernel_size=3))

        self.HG2_dowsample_0 = dowsample(embed_dim, embed_dim * 2)
        HG2_res_enc_1 = [
            common.ResBlock(
                conv, embed_dim * 2, kernel_size=3
            ) for _ in range(n_resblocks)
        ]
        self.HG2_res_enc1 = nn.Sequential(*HG2_res_enc_1)
        self.HG2_encoderlayer_1 = BasicBaseModelLayer(dim=embed_dim * 2,
                                                    output_dim=embed_dim * 2,
                                                    input_resolution=(img_size // 2,
                                                                      img_size // 2),
                                                    depth=depths[1],
                                                    num_heads=num_heads[1],
                                                    win_size=win_size,
                                                    mlp_ratio=self.mlp_ratio,
                                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                                    drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                                                    norm_layer=norm_layer,
                                                    use_checkpoint=use_checkpoint,
                                                    token_projection=token_projection, token_mlp=token_mlp,
                                                    se_layer=se_layer)
        self.HG2_conv_enc1 = nn.Sequential(conv(embed_dim * 2, embed_dim * 2, kernel_size=3))

        self.HG2_dowsample_1 = dowsample(embed_dim * 2, embed_dim * 4)

        # Bottleneck
        self.conv_HG2 = BasicBaseModelLayer(dim=embed_dim * 4,
                                          output_dim=embed_dim * 4,
                                          input_resolution=(img_size // (2 ** 2),
                                                            img_size // (2 ** 2)),
                                          depth=depths[4],
                                          num_heads=num_heads[4],
                                          win_size=win_size,
                                          mlp_ratio=self.mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          drop=drop_rate, attn_drop=attn_drop_rate,
                                          drop_path=conv_dpr,
                                          norm_layer=norm_layer,
                                          use_checkpoint=use_checkpoint,
                                          token_projection=token_projection, token_mlp=token_mlp, se_layer=se_layer)

        # Decoder
        self.HG2_upsample_0 = upsample(embed_dim * 4, embed_dim * 2)
        HG2_res_dec_0 = [
            common.ResBlock(
                conv, embed_dim * 4, kernel_size=3
            ) for _ in range(n_resblocks)
        ]
        self.HG2_res_dec0 = nn.Sequential(*HG2_res_dec_0)
        self.HG2_decoderlayer_0 = BasicBaseModelLayer(dim=embed_dim * 4,
                                                    output_dim=embed_dim * 4,
                                                    input_resolution=(img_size // 2,
                                                                      img_size // 2),
                                                    depth=depths[5],
                                                    num_heads=num_heads[5],
                                                    win_size=win_size,
                                                    mlp_ratio=self.mlp_ratio,
                                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                                    drop_path=dec_dpr[:depths[5]],
                                                    norm_layer=norm_layer,
                                                    use_checkpoint=use_checkpoint,
                                                    token_projection=token_projection, token_mlp=token_mlp,
                                                    se_layer=se_layer)

        self.HG2_conv_dec0 = nn.Sequential(conv(embed_dim * 4, embed_dim * 4, kernel_size=3))

        self.HG2_upsample_1 = upsample(embed_dim * 4, embed_dim)
        HG2_res_dec_1 = [
            common.ResBlock(
                conv, embed_dim * 2, kernel_size=3
            ) for _ in range(n_resblocks)
        ]
        self.HG2_res_dec1 = nn.Sequential(*HG2_res_dec_1)
        self.HG2_decoderlayer_1 = BasicBaseModelLayer(dim=embed_dim * 2,
                                                    output_dim=embed_dim * 2,
                                                    input_resolution=(img_size,
                                                                      img_size),
                                                    depth=depths[6],
                                                    num_heads=num_heads[6],
                                                    win_size=win_size,
                                                    mlp_ratio=self.mlp_ratio,
                                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                    drop=drop_rate, attn_drop=attn_drop_rate,
                                                    drop_path=dec_dpr[sum(depths[5:6]):sum(depths[5:7])],
                                                    norm_layer=norm_layer,
                                                    use_checkpoint=use_checkpoint,
                                                    token_projection=token_projection, token_mlp=token_mlp,
                                                    se_layer=se_layer)
        self.HG2_conv_dec1 = nn.Sequential(conv(embed_dim * 2, embed_dim * 2, kernel_size=3))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, token_projection={self.token_projection}, token_mlp={self.mlp},win_size={self.win_size}"

    def forward(self, x, mask=None):
        # Input Multi-Frame Conv
        b, t, c, h, w = x.size()
        assert c == 3, 'In channels should be 3!'

        x_base = x[:, 0, :, :, :].contiguous()

        ### feature extraction of aligned frames
        x_feat_head = self.head(x.view(-1, c, h, w))  # [b*t, embed_dim, h, w]
        x_feat_body = self.body(x_feat_head)  # [b*t, embed_dim, h, w]

        feat = x_feat_body.view(b, t, -1, h, w)   # [b, t, embed_dim, h, w]

        ### fusion of aligned features
        fusion_feat = self.fusion(feat)   # fusion feat [b, embed_dim, h, w]

        assert fusion_feat.dim() == 4, 'Fusion Feat should be [B,C,H,W]!'

        # Input Projection
        y = self.input_proj(fusion_feat)   # B, H*W, C
        y = self.pos_drop(y)

        #### HG1 #####
        #Encoder
        conv0 = self.HG1_encoderlayer_0(y,mask=mask)
        pool0 = self.HG1_dowsample_0(conv0)

        conv1 = self.HG1_encoderlayer_1(pool0,mask=mask)
        pool1 = self.HG1_dowsample_1(conv1)

        # Bottleneck
        conv2 = self.conv_HG1(pool1, mask=mask)

        #Decoder
        up0 = self.HG1_upsample_0(conv2)
        deconv0 = torch.cat([up0,conv1],-1)
        deconv0 = self.HG1_decoderlayer_0(deconv0,mask=mask)

        up1 = self.HG1_upsample_1(deconv0)
        deconv1 = torch.cat([up1,conv0],-1)
        deconv1 = self.HG1_decoderlayer_1(deconv1,mask=mask)

        # Output Projection
        y_1 = self.output_proj(deconv1)

        #### HG2 #####
        # Encoder
        conv0_2 = self.HG2_encoderlayer_0(y_1, mask=mask)
        pool0_2 = self.HG2_dowsample_0(conv0_2)

        conv1_2 = self.HG2_encoderlayer_1(pool0_2, mask=mask)
        pool1_2 = self.HG2_dowsample_1(conv1_2)

        # Bottleneck
        conv2_2 = self.conv_HG2(pool1_2, mask=mask)

        # Decoder
        up0_2 = self.HG2_upsample_0(conv2_2)
        deconv0_2 = self.output_proj_HG2_0(torch.cat([up0, conv1, up0_2, conv1_2], -1))  # B, H/2*W/2, C*8
        deconv0_2 = self.HG2_decoderlayer_0(deconv0_2, mask=mask)

        up1_2 = self.HG2_upsample_1(deconv0_2)
        deconv1_2 = self.output_proj_HG2_1(torch.cat([up1, conv0, up1_2, conv0_2], -1))
        deconv1_2 = self.HG2_decoderlayer_1(deconv1_2, mask=mask)

        # Output Projection
        y_2 = self.output_proj_2(deconv1_2)

        output_2 = self.tail(y_2)

        base = F.interpolate(x_base, scale_factor=4, mode='bilinear', align_corners=False)

        out = output_2 + base

        return out

    def flops(self):
        flops = 0
        # Input Projection
        flops += self.input_proj.flops(self.reso,self.reso)
        # Encoder
        flops += self.encoderlayer_0.flops()+self.dowsample_0.flops(self.reso,self.reso)
        flops += self.encoderlayer_1.flops()+self.dowsample_1.flops(self.reso//2,self.reso//2)
        flops += self.encoderlayer_2.flops()+self.dowsample_2.flops(self.reso//2**2,self.reso//2**2)
        flops += self.encoderlayer_3.flops()+self.dowsample_3.flops(self.reso//2**3,self.reso//2**3)

        # Bottleneck
        flops += self.conv.flops()

        # Decoder
        flops += self.upsample_0.flops(self.reso//2**4,self.reso//2**4)+self.decoderlayer_0.flops()
        flops += self.upsample_1.flops(self.reso//2**3,self.reso//2**3)+self.decoderlayer_1.flops()
        flops += self.upsample_2.flops(self.reso//2**2,self.reso//2**2)+self.decoderlayer_2.flops()
        flops += self.upsample_3.flops(self.reso//2,self.reso//2)+self.decoderlayer_3.flops()

        # Output Projection
        flops += self.output_proj.flops(self.reso,self.reso)
        return flops
