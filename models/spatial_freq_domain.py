import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops


# ==============================================================================
# 1. 空间注意力
# ==============================================================================
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, 3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        att = torch.cat([avg_pool, max_pool], dim=1)
        att = self.conv(att)
        return x * att


# ==============================================================================
# 2. SAFM 融合模块
# ==============================================================================
class SAFM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.spatial_att = SpatialAttention()
        self.weight = nn.Sequential(
            nn.Conv2d(dim * 2, 2, 1, bias=False),
            nn.Softmax(dim=1)
        )

    def forward(self, f_plain, f_four):
        f_plain = self.spatial_att(f_plain)
        f_four = self.spatial_att(f_four)
        weight = self.weight(torch.cat([f_plain, f_four], dim=1))
        w1 = weight[:, 0:1, :, :]
        w2 = weight[:, 1:2, :, :]
        fused = w1 * f_plain + w2 * f_four
        return fused


# ==============================================================================
# 3. 条形卷积
# ==============================================================================
class StripConv(nn.Module):
    def __init__(self, in_c, out_c, direction):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False)
        kernel = torch.zeros_like(self.conv.weight.data)
        c = out_c
        if direction == 'h':
            kernel[:, :, 1, :] = 1.0
        elif direction == 'v':
            kernel[:, :, :, 1] = 1.0
        elif direction == 'd1':
            kernel[:, :, 0, 2] = kernel[:, :, 1, 1] = kernel[:, :, 2, 0] = 1.0
        elif direction == 'd2':
            kernel[:, :, 0, 0] = kernel[:, :, 1, 1] = kernel[:, :, 2, 2] = 1.0
        self.conv.weight.data = kernel
        self.conv.weight.requires_grad_(False)

    def forward(self, x):
        return self.conv(x)


# ==============================================================================
# 4. 四分支模块（strip + c/w/h注意力）
# ==============================================================================
class FourBranchModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        mid = dim // 4

        self.strip = StripConv(dim, mid, 'h')
        self.ca_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )
        self.ca_conv = nn.Conv2d(dim, mid, 1)
        self.wa_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Conv2d(dim, dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )
        self.wa_conv = nn.Conv2d(dim, mid, 1)
        self.ha_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((None, 1)),
            nn.Conv2d(dim, dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )
        self.ha_conv = nn.Conv2d(dim, mid, 1)
        self.four_out = nn.Sequential(
            nn.Conv2d(mid * 4, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b_strip = self.strip(x)
        ca_weight = self.ca_branch(x)
        b_c = x * ca_weight
        b_c = self.ca_conv(b_c)
        wa_weight = self.wa_branch(x)
        b_w = x * wa_weight
        b_w = self.wa_conv(b_w)
        ha_weight = self.ha_branch(x)
        b_h = x * ha_weight
        b_h = self.ha_conv(b_h)
        fuse = torch.cat([b_strip, b_c, b_w, b_h], dim=1)
        return self.four_out(fuse)


# ==============================================================================
# 5. 最终模块：普通卷积 + 四分支模块 + SAFM融合
# ==============================================================================
class Plain_Four_SAFM_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.plain_branch = nn.Conv2d(dim, dim, 3, padding=1)
        self.four_branch = FourBranchModule(dim)
        self.safm = SAFM(dim)
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        f_plain = self.plain_branch(x)
        f_four = self.four_branch(x)
        out = self.safm(f_plain, f_four)
        out = self.norm(out)
        out = self.act(out)
        return out


class VesselPriorExtractor(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()

        sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32)

        self.sobel_x = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False)
        self.sobel_y = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False)

        self.sobel_x.weight = nn.Parameter(sobel_x.repeat(in_channels,1,1,1), requires_grad=False)
        self.sobel_y.weight = nn.Parameter(sobel_y.repeat(in_channels,1,1,1), requires_grad=False)

        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

        self.out_conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        sx = torch.abs(self.sobel_x(x))
        sy = torch.abs(self.sobel_y(x))
        edge = sx + sy

        avg = torch.mean(edge, dim=1, keepdim=True)
        maxv, _ = torch.max(edge, dim=1, keepdim=True)
        att = self.spatial_att(torch.cat([avg, maxv], dim=1))
        prior = edge * att

        F_prior = self.out_conv(prior)

        return F_prior


class VesselPriorExtractorSobelOnly(nn.Module):
    """Sobel-only 先验：去掉空间注意力，edge 直接做通道投影"""

    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

        self.sobel_x = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False)
        self.sobel_y = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False)

        self.sobel_x.weight = nn.Parameter(sobel_x.repeat(in_channels, 1, 1, 1), requires_grad=False)
        self.sobel_y.weight = nn.Parameter(sobel_y.repeat(in_channels, 1, 1, 1), requires_grad=False)

        self.out_conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        sx = torch.abs(self.sobel_x(x))
        sy = torch.abs(self.sobel_y(x))
        edge = sx + sy

        F_prior = self.out_conv(edge)

        return F_prior


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, pad=0, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=bias)
    def forward(self, x):
        return F.relu(self.conv(x), inplace=True)


class conv_relu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super(conv_relu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=True, dilation=dilation_rate),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_input):
        out = self.conv(x_input)
        return out


class conv_leakyrelu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super(conv_leakyrelu, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                      padding=padding, bias=True, dilation=dilation_rate),
            nn.LeakyReLU(negative_slope=0.01, inplace=False)
        )

    def forward(self, x_input):
        out = self.conv(x_input)
        return out


class conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, dilation_rate=1, padding=0, stride=1):
        super(conv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=True, dilation=dilation_rate)

    def forward(self, x_input):
        out = self.conv(x_input)
        return out


def block_images_einops(x, patch_size):
    batch, height, width, channels = x.shape
    grid_height = height // patch_size[0]
    grid_width = width // patch_size[1]
    x = einops.rearrange(
        x, "n (gh fh) (gw fw) c -> n (gh gw) (fh fw) c",
        gh=grid_height, gw=grid_width, fh=patch_size[0], fw=patch_size[1])
    return x


def unblock_images_einops(x, grid_size, patch_size):
    x = einops.rearrange(
        x, "n (gh gw) (fh fw) c -> n (gh fh) (gw fw) c",
        gh=grid_size[0], gw=grid_size[1], fh=patch_size[0], fw=patch_size[1])
    return x


class Layer_norm_process(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.c = c
        self.beta = nn.Parameter(torch.zeros(c))
        self.gamma = nn.Parameter(torch.ones(c))

    def forward(self, feature):
        var_mean = torch.var_mean(feature, dim=-1, unbiased=False)
        mean = var_mean[1]
        var = var_mean[0]
        feature = (feature - mean[..., None]) / torch.sqrt(var[..., None] + self.eps)
        gamma = self.gamma.expand_as(feature)
        beta = self.beta.expand_as(feature)
        feature = feature * gamma + beta
        return feature


class CrossGating(nn.Module):
    def __init__(self, in_channel, block_size, grid_size, input_proj_factor=2, dropout_rate=0.0, bias=True,
                 gate_type='full'):
        super().__init__()
        self.in_channel = in_channel
        self.block_size = block_size
        self.grid_size = grid_size
        self.gate_type = gate_type
        self.gh = self.grid_size[0]
        self.gw = self.grid_size[1]
        self.fh = self.block_size[0]
        self.fw = self.block_size[1]

        self.input_proj_factor = input_proj_factor
        self.dropout_rate = dropout_rate
        self.bias = bias
        self.Dense_0 = nn.Linear(self.gh*self.gw, self.gh*self.gw, bias = self.bias)
        self.Dense_1 = nn.Linear(self.fh*self.fw, self.fh*self.fw, bias = self.bias)
        self.out_project = nn.Linear(self.in_channel*self.input_proj_factor, self.in_channel, bias=self.bias)
        self.in_project = nn.Linear(self.in_channel, self.in_channel*self.input_proj_factor, bias=self.bias)
        self.gelu = nn.GELU()

        self.LayerNorm_in = Layer_norm_process(self.in_channel)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        _, h, w, _ = x.shape

        x = self.LayerNorm_in(x)
        x = self.in_project(x)
        x = self.gelu(x)
        c = x.size(-1)//2
        u, v = torch.split(x, c, dim=-1)

        if self.gate_type in ('full', 'grid_only'):
            fh, fw = h//self.gh, w//self.gw
            u = block_images_einops(u, patch_size = (fh, fw))
            u = u.permute(0,3,2,1)
            u = self.Dense_0(u)
            u = u.permute(0,3,2,1)
            u = unblock_images_einops(u, grid_size=(self.gh, self.gw), patch_size=(fh, fw))

        if self.gate_type in ('full', 'block_only'):
            gh, gw = h//self.fh, w//self.fw
            v = block_images_einops(v, patch_size=(self.fh, self.fw))
            v = v.permute(0,1,3,2)
            v = self.Dense_1(v)
            v = v.permute(0,1,3,2)
            v = unblock_images_einops(v, grid_size=(gh, gw), patch_size=(self.fh, self.fw))

        x = torch.cat([u,v], dim=-1)
        x = self.out_project(x)
        x = self.dropout(x)

        return x


class RDB(nn.Module):
    def __init__(self, in_channel, d_list, inter_num):
        super(RDB, self).__init__()
        self.d_list = d_list
        self.conv_layers = nn.ModuleList()
        c = in_channel
        for i in range(len(d_list)):
            dense_conv = conv_relu(in_channel=c, out_channel=inter_num, kernel_size=3, dilation_rate=d_list[i],
                                   padding=d_list[i])
            self.conv_layers.append(dense_conv)
            c = c + inter_num
        self.conv_post = conv(in_channel=c, out_channel=in_channel, kernel_size=1)

    def forward(self, x):
        t = x
        for conv_layer in self.conv_layers:
            _t = conv_layer(t)
            t = torch.cat([_t, t], dim=1)
        t = self.conv_post(t)
        return t


class Cross_Attention_SpatialFreq(nn.Module):
    def __init__(self, in_channel, features, grid_size, block_size, dropout_rate=0.0,
                 input_proj_factor=2, bias=True, inter_num=32, gate_type='full'):
        super().__init__()
        self.in_channel = in_channel
        self.features = features
        self.grid_size = grid_size
        self.block_size = block_size
        self.dropout_rate = dropout_rate
        self.input_proj_factor = input_proj_factor
        self.bias = bias
        self.inter_num = inter_num
        self.gate_type = gate_type
        
        self.conv1_spatial = Conv1x1(self.in_channel, self.features, bias=self.bias)
        self.conv1_freq_in = Conv1x1(self.in_channel, self.features, bias=self.bias)
        self.conv1_freq_out = Conv1x1(self.features, self.in_channel, bias=self.bias)
        
        self.in_linear_spatial = nn.Linear(self.features, self.features, bias=self.bias)
        self.in_linear_freq = nn.Linear(self.features, self.features, bias=self.bias)
        self.out_linear_spatial = nn.Linear(self.features, self.features, bias=self.bias)
        self.out_linear_freq = nn.Linear(self.features, self.features, bias=self.bias)
        
        self.getspatialgatingweights_spatial = CrossGating(
            in_channel=self.features,
            block_size=self.block_size,
            grid_size=self.grid_size,
            dropout_rate=self.dropout_rate,
            bias=self.bias,
            gate_type=self.gate_type)
        
        self.getspatialgatingweights_freq = CrossGating(
            in_channel=self.features,
            block_size=self.block_size,
            grid_size=self.grid_size,
            dropout_rate=self.dropout_rate,
            bias=self.bias,
            gate_type=self.gate_type)

        self.LayerNorm_spatial = Layer_norm_process(self.features)
        self.LayerNorm_freq = Layer_norm_process(self.features)

        self.gelu1 = nn.GELU()
        self.gelu2 = nn.GELU()
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.dropout2 = nn.Dropout(self.dropout_rate)

        self.rdb_spatial = RDB(in_channel=in_channel, d_list=(1, 2, 1), inter_num=self.inter_num)
        self.rdb_freq = RDB(in_channel=in_channel, d_list=(1, 2, 1), inter_num=self.inter_num)

    def forward(self, spatial_x, freq_x):
        res_spatial = spatial_x
        res_freq = freq_x

        spatial_x = self.conv1_spatial(spatial_x)
        freq_x = self.conv1_freq_in(freq_x)

        spatial_x = self.rdb_spatial(spatial_x)
        freq_x = self.rdb_freq(freq_x)

        spatial_x = spatial_x.permute(0,2,3,1)
        freq_x = freq_x.permute(0,2,3,1)

        if freq_x.shape != spatial_x.shape:
            freq_x = F.interpolate(freq_x.permute(0,3,1,2), size=spatial_x.shape[1:3], mode='bilinear').permute(0,2,3,1)
        
        spatial_x = self.LayerNorm_spatial(spatial_x)
        spatial_x = self.in_linear_spatial(spatial_x)
        spatial_x = self.gelu1(spatial_x)
        g_spatial = self.getspatialgatingweights_spatial(spatial_x)

        freq_x = self.LayerNorm_freq(freq_x)
        freq_x = self.in_linear_freq(freq_x)
        freq_x = self.gelu2(freq_x)
        g_freq = self.getspatialgatingweights_freq(freq_x)

        freq_x = freq_x * g_spatial
        freq_x = self.out_linear_freq(freq_x)
        freq_x = self.dropout1(freq_x)

        spatial_x = spatial_x * g_freq
        spatial_x = self.out_linear_spatial(spatial_x)
        spatial_x = self.dropout2(spatial_x)

        spatial_x = spatial_x.permute(0,3,1,2)
        freq_x = freq_x.permute(0,3,1,2)

        if freq_x.shape[2:] != res_freq.shape[2:]:
            freq_x = F.interpolate(freq_x, size=res_freq.shape[2:], mode='bilinear')
        freq_x = self.conv1_freq_out(freq_x)

        spatial_x = spatial_x + res_spatial
        freq_x = freq_x + res_freq

        return spatial_x, freq_x


def get_wav(in_channels):
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    net = nn.Conv2d
   
    LL = net(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False, groups=in_channels)
    LH = net(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False, groups=in_channels)
    HL = net(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False, groups=in_channels)
    HH = net(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False, groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
    return LL, LH, HL, HH


class Wavelet(nn.Module):
    def __init__(self, in_channels):
        super(Wavelet, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)


class SpatialBranch(nn.Module):
    def __init__(self, in_dim, mid_dim=None, kernel_size=3):
        super().__init__()
        if mid_dim is None:
            mid_dim = in_dim
        
        self.in_proj = nn.Conv2d(in_dim, mid_dim, kernel_size=1)
        
        self.horizontal_conv = nn.Conv2d(
            mid_dim, mid_dim,
            kernel_size=(1, kernel_size),
            stride=1,
            padding=(0, kernel_size//2),
            groups=mid_dim
        )
        
        self.vertical_conv = nn.Conv2d(
            mid_dim, mid_dim,
            kernel_size=(kernel_size, 1),
            stride=1,
            padding=(kernel_size//2, 0),
            groups=mid_dim
        )
        
        self.out_proj = nn.Conv2d(mid_dim * 2, in_dim, kernel_size=1)
        self.norm = nn.GroupNorm(in_dim // 8, in_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.in_proj(x)
        
        x_h = self.horizontal_conv(x)
        x_v = self.vertical_conv(x)
        
        x = torch.cat([x_h, x_v], dim=1)
        x = self.out_proj(x)
        x = self.norm(x + residual)
        x = self.activation(x)
        return x


class FrequencyBranch(nn.Module):
    def __init__(self, in_dim, mid_dim=None):
        super().__init__()
        if mid_dim is None:
            mid_dim = in_dim // 2
        
        self.wavelet = Wavelet(in_dim)
        
        self.ll_conv = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, kernel_size=1),
            nn.GroupNorm(mid_dim // 8, mid_dim),
            nn.GELU()
        )
        self.lh_conv = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, kernel_size=1),
            nn.GroupNorm(mid_dim // 8, mid_dim),
            nn.GELU()
        )
        self.hl_conv = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, kernel_size=1),
            nn.GroupNorm(mid_dim // 8, mid_dim),
            nn.GELU()
        )
        self.hh_conv = nn.Sequential(
            nn.Conv2d(in_dim, mid_dim, kernel_size=1),
            nn.GroupNorm(mid_dim // 8, mid_dim),
            nn.GELU()
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(mid_dim * 4, in_dim, kernel_size=1),
            nn.GroupNorm(in_dim // 8, in_dim),
            nn.GELU()
        )
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        B, C, H, W = x.shape
        
        LL, LH, HL, HH = self.wavelet(x)
        
        ll_feat = self.ll_conv(LL)
        lh_feat = self.lh_conv(LH)
        hl_feat = self.hl_conv(HL)
        hh_feat = self.hh_conv(HH)
        
        ll_feat = self.upsample(ll_feat)
        lh_feat = self.upsample(lh_feat)
        hl_feat = self.upsample(hl_feat)
        hh_feat = self.upsample(hh_feat)
        
        if ll_feat.shape[2:] != (H, W):
            ll_feat = F.interpolate(ll_feat, size=(H, W), mode='bilinear', align_corners=True)
            lh_feat = F.interpolate(lh_feat, size=(H, W), mode='bilinear', align_corners=True)
            hl_feat = F.interpolate(hl_feat, size=(H, W), mode='bilinear', align_corners=True)
            hh_feat = F.interpolate(hh_feat, size=(H, W), mode='bilinear', align_corners=True)
        
        x = torch.cat([ll_feat, lh_feat, hl_feat, hh_feat], dim=1)
        x = self.fusion(x)
        return x


class SpatialFreqDomainBlock(nn.Module):
    def __init__(self, in_dim, spatial_kernel=3, freq_mid_dim=None, grid_size=(2, 2), block_size=(4, 4),
                 gate_type='full'):
        super().__init__()
        
        self.spatial_branch = SpatialBranch(in_dim, kernel_size=spatial_kernel)
        self.freq_branch = FrequencyBranch(in_dim, mid_dim=freq_mid_dim)
        
        self.cross_attention = Cross_Attention_SpatialFreq(
            in_channel=in_dim,
            features=in_dim,
            grid_size=grid_size,
            block_size=block_size,
            gate_type=gate_type
        )
        
        self.final_fusion = nn.Sequential(
            nn.Conv2d(in_dim * 2, in_dim, kernel_size=3, padding=1),
            nn.GroupNorm(in_dim // 8, in_dim),
            nn.GELU()
        )

    def forward(self, x):
        residual = x
        
        spatial_feat = self.spatial_branch(x)
        freq_feat = self.freq_branch(x)
        
        spatial_feat, freq_feat = self.cross_attention(spatial_feat, freq_feat)
        
        combined = torch.cat([spatial_feat, freq_feat], dim=1)
        x = self.final_fusion(combined)
        x = x + residual
        
        return x


# ===================== 差分CNN分支 DiffNet =====================
class DiffNet(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 可学习的差分卷积（不用Sobel，避免重复）
        self.conv_center = nn.Conv2d(channels, channels, 1)  # 中心特征
        self.conv_surround = nn.Conv2d(channels, channels, 3, padding=1)  # 周围特征
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        # 差分运算：中心特征 - 周围平均特征
        center_feat = self.conv_center(x)
        surround_feat = self.conv_surround(x)
        diff_feat = center_feat - surround_feat
        
        out = self.act(self.bn1(self.conv1(diff_feat)))
        return out + residual

# ===================== 边缘增强通道自注意力 EdgeCSA =====================
class EdgeCSA(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1, groups=channels)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        
        # 动态调整kernel_size
        kernel_size = min(self.kernel_size, H, W)
        stride = kernel_size
        
        # 混合池化
        x_avg = F.avg_pool2d(x, kernel_size, stride)
        x_max = F.max_pool2d(x, kernel_size, stride)
        x_compress = (x_avg + x_max) / 2
        
        # QKV自注意力
        q, k, v = self.qkv(x_compress).chunk(3, 1)
        q = q.flatten(2).transpose(1, 2)
        k = k.flatten(2)
        v = v.flatten(2).transpose(1, 2)
        attn = (q @ k) * (self.channels ** -0.5)
        attn = attn.softmax(dim=-1)
        out = attn @ v
        
        # 尺寸恢复
        out_h = H // stride
        out_w = W // stride
        out = out.transpose(1, 2).reshape(B, C, out_h, out_w)
        out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        
        # 归一化
        out = out.permute(0, 2, 3, 1)
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2)
        
        # 注意力加权
        attn_weight = self.sigmoid(self.proj(out))
        return residual * attn_weight

# ===================== 解码器并行模块 =====================
class DecoderParallelBlock(nn.Module):
    """LGF 解码并行模块。

    branch_type:
        'full'        局部差分 CNN 分支 + 全局通道自注意力分支（默认）
        'local_only'  只保留 differential CNN local branch
        'global_only' 只保留 channel self-attention global branch
    """

    def __init__(self, channels, branch_type='full'):
        super().__init__()
        self.branch_type = branch_type
        if branch_type == 'full':
            self.diff_branch = DiffNet(channels)
            self.csa_branch = EdgeCSA(channels)
            self.fusion_conv = nn.Conv2d(channels * 2, channels, 1)
        elif branch_type == 'local_only':
            self.diff_branch = DiffNet(channels)
            self.fusion_conv = nn.Conv2d(channels, channels, 1)
        elif branch_type == 'global_only':
            self.csa_branch = EdgeCSA(channels)
            self.fusion_conv = nn.Conv2d(channels, channels, 1)
        else:
            raise ValueError(f"Unknown branch_type: {branch_type}")
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        if self.branch_type == 'full':
            diff_out = self.diff_branch(x)
            csa_out = self.csa_branch(x)
            fused = torch.cat([diff_out, csa_out], dim=1)
        elif self.branch_type == 'local_only':
            fused = self.diff_branch(x)
        elif self.branch_type == 'global_only':
            fused = self.csa_branch(x)
        out = self.act(self.bn(self.fusion_conv(fused)))
        return out

class MultiBranchSpatialFreq(nn.Module):
    def __init__(self, in_dim, spatial_kernel=3, freq_mid_dim=None, num_blocks=1, grid_size=(2, 2), block_size=(4, 4),
                 gate_type='full'):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            SpatialFreqDomainBlock(in_dim, spatial_kernel, freq_mid_dim, grid_size, block_size, gate_type)
            for _ in range(num_blocks)
        ])
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1),
            nn.GroupNorm(in_dim // 8, in_dim),
            nn.GELU()
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.final_conv(x)
        return x


class ConcatFusionBlock(nn.Module):
    """MBSFF 消融：空间+频率分支不变，用 Concatenation（cat -> 1x1 conv）替换 cross-gating"""

    def __init__(self, in_dim, spatial_kernel=3, freq_mid_dim=None):
        super().__init__()
        self.spatial_branch = SpatialBranch(in_dim, kernel_size=spatial_kernel)
        self.freq_branch = FrequencyBranch(in_dim, mid_dim=freq_mid_dim)
        self.fusion = nn.Sequential(
            nn.Conv2d(in_dim * 2, in_dim, kernel_size=1),
            nn.GroupNorm(in_dim // 8, in_dim),
            nn.GELU()
        )

    def forward(self, x):
        residual = x
        spatial_feat = self.spatial_branch(x)
        freq_feat = self.freq_branch(x)
        combined = torch.cat([spatial_feat, freq_feat], dim=1)
        x = self.fusion(combined)
        x = x + residual
        return x


class AdditionFusionBlock(nn.Module):
    """MBSFF 消融：空间+频率分支不变，用 Addition（F_spatial + F_freq）替换 cross-gating"""

    def __init__(self, in_dim, spatial_kernel=3, freq_mid_dim=None):
        super().__init__()
        self.spatial_branch = SpatialBranch(in_dim, kernel_size=spatial_kernel)
        self.freq_branch = FrequencyBranch(in_dim, mid_dim=freq_mid_dim)
        self.fusion = nn.Sequential(
            nn.GroupNorm(in_dim // 8, in_dim),
            nn.GELU()
        )

    def forward(self, x):
        residual = x
        spatial_feat = self.spatial_branch(x)
        freq_feat = self.freq_branch(x)
        combined = spatial_feat + freq_feat
        x = self.fusion(combined)
        x = x + residual
        return x


class MultiBranchConcatFusion(nn.Module):
    """MBSFF 消融：Concatenation 交互的 MultiBranch 容器"""

    def __init__(self, in_dim, spatial_kernel=3, freq_mid_dim=None, num_blocks=1):
        super().__init__()
        self.blocks = nn.ModuleList([
            ConcatFusionBlock(in_dim, spatial_kernel, freq_mid_dim)
            for _ in range(num_blocks)
        ])
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1),
            nn.GroupNorm(in_dim // 8, in_dim),
            nn.GELU()
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.final_conv(x)
        return x


class MultiBranchAdditionFusion(nn.Module):
    """MBSFF 消融：Addition 交互的 MultiBranch 容器"""

    def __init__(self, in_dim, spatial_kernel=3, freq_mid_dim=None, num_blocks=1):
        super().__init__()
        self.blocks = nn.ModuleList([
            AdditionFusionBlock(in_dim, spatial_kernel, freq_mid_dim)
            for _ in range(num_blocks)
        ])
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1),
            nn.GroupNorm(in_dim // 8, in_dim),
            nn.GELU()
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.final_conv(x)
        return x


class SpatialOnlyBlock(nn.Module):
    """MBSFF 消融：只保留空间分支（无频率分支、无 cross-gating）"""

    def __init__(self, in_dim, spatial_kernel=3):
        super().__init__()
        self.spatial_branch = SpatialBranch(in_dim, kernel_size=spatial_kernel)
        self.final_fusion = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1),
            nn.GroupNorm(in_dim // 8, in_dim),
            nn.GELU()
        )

    def forward(self, x):
        residual = x
        spatial_feat = self.spatial_branch(x)
        x = self.final_fusion(spatial_feat)
        x = x + residual
        return x


class FrequencyOnlyBlock(nn.Module):
    """MBSFF 消融：只保留 Haar 频率分支（无空间分支、无 cross-gating）"""

    def __init__(self, in_dim, freq_mid_dim=None):
        super().__init__()
        self.freq_branch = FrequencyBranch(in_dim, mid_dim=freq_mid_dim)
        self.final_fusion = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1),
            nn.GroupNorm(in_dim // 8, in_dim),
            nn.GELU()
        )

    def forward(self, x):
        residual = x
        freq_feat = self.freq_branch(x)
        x = self.final_fusion(freq_feat)
        x = x + residual
        return x


class MultiBranchSpatialOnly(nn.Module):
    """MBSFF 消融：仅空间分支的 MultiBranch 容器（对齐 MultiBranchSpatialFreq 结构）"""

    def __init__(self, in_dim, spatial_kernel=3, num_blocks=1):
        super().__init__()
        self.blocks = nn.ModuleList([
            SpatialOnlyBlock(in_dim, spatial_kernel)
            for _ in range(num_blocks)
        ])
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1),
            nn.GroupNorm(in_dim // 8, in_dim),
            nn.GELU()
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.final_conv(x)
        return x


class MultiBranchFrequencyOnly(nn.Module):
    """MBSFF 消融：仅频率分支的 MultiBranch 容器（对齐 MultiBranchSpatialFreq 结构）"""

    def __init__(self, in_dim, freq_mid_dim=None, num_blocks=1):
        super().__init__()
        self.blocks = nn.ModuleList([
            FrequencyOnlyBlock(in_dim, freq_mid_dim)
            for _ in range(num_blocks)
        ])
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1),
            nn.GroupNorm(in_dim // 8, in_dim),
            nn.GELU()
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.final_conv(x)
        return x
