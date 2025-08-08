import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch.utils.checkpoint as checkpoint

class DepthwiseSeparableConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super(DepthwiseSeparableConv3D, self).__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class WindowedSelfAttention3D(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=4, dropout=0.1):
        super(WindowedSelfAttention3D, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim
        self.window_size = window_size

        self.query = nn.Conv3d(dim, dim, kernel_size=1)
        self.key = nn.Conv3d(dim, dim, kernel_size=1)
        self.value = nn.Conv3d(dim, dim, kernel_size=1)
        self.proj = nn.Conv3d(dim, dim, kernel_size=1)
        self.scale = self.head_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, D, H, W = x.shape
        window = self.window_size

        pad_d_total = (window - D % window) if D % window != 0 else 0
        pad_h_total = (window - H % window) if H % window != 0 else 0
        pad_w_total = (window - W % window) if W % window != 0 else 0

        pad_d_front = pad_d_total // 2
        pad_d_back = pad_d_total - pad_d_front
        pad_h_left = pad_h_total // 2
        pad_h_right = pad_h_total - pad_h_left
        pad_w_left = pad_w_total // 2
        pad_w_right = pad_w_total - pad_w_left

        if pad_d_total > 0 or pad_h_total > 0 or pad_w_total > 0:
            x = F.pad(x, (pad_w_left, pad_w_right, pad_h_left, pad_h_right, pad_d_front, pad_d_back))
            D_padded, H_padded, W_padded = x.size(2), x.size(3), x.size(4)
        else:
            D_padded, H_padded, W_padded = D, H, W

        x = x.unfold(2, window, window).unfold(3, window, window).unfold(4, window, window)
        B, C, num_windows_d, num_windows_h, num_windows_w, window_d, window_h, window_w = x.shape
        x = x.contiguous().view(B * num_windows_d * num_windows_h * num_windows_w, C, window_d, window_h, window_w)

        q = self.query(x).reshape(B * num_windows_d * num_windows_h * num_windows_w, self.num_heads, self.head_dim, -1)
        k = self.key(x).reshape(B * num_windows_d * num_windows_h * num_windows_w, self.num_heads, self.head_dim, -1)
        v = self.value(x).reshape(B * num_windows_d * num_windows_h * num_windows_w, self.num_heads, self.head_dim, -1)

        attn = torch.matmul(q.transpose(-2, -1), k) * self.scale
        attn = self.softmax(attn)
        attn = self.dropout(attn)

        out = torch.matmul(v, attn.transpose(-2, -1))
        out = out.reshape(B * num_windows_d * num_windows_h * num_windows_w, C, window_d, window_h, window_w)
        out = self.proj(out)

        out = out.view(B, num_windows_d, num_windows_h, num_windows_w, C, window_d, window_h, window_w)
        out = out.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        out = out.view(B, C, D_padded, H_padded, W_padded)

        if pad_d_total > 0 or pad_h_total > 0 or pad_w_total > 0:
            out = out[:, :, pad_d_front:D_padded - pad_d_back, pad_h_left:H_padded - pad_h_right, pad_w_left:W_padded - pad_w_right]

        return out

class PyramidPooling3D(nn.Module):
    def __init__(self, in_channels, pool_sizes=(1, 2, 3, 6), mode='avg'):
        super(PyramidPooling3D, self).__init__()
        self.pool_sizes = pool_sizes
        self.mode = mode
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool3d(output_size=(size, size, size)),
                nn.Conv3d(in_channels, in_channels // len(pool_sizes), kernel_size=1, bias=False),
                nn.ReLU(inplace=True)
            ) for size in pool_sizes
        ])
        # 将 InstanceNorm3d 移到插值之后
        self.bn_relu = nn.Sequential(
            nn.InstanceNorm3d(in_channels // len(pool_sizes), affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        D, H, W = x.size(2), x.size(3), x.size(4)
        pyramids = [x]
        for stage in self.stages:
            out = stage(x)  # (B, reduced_channels, size, size, size)
            out = F.interpolate(out, size=(D, H, W), mode='trilinear', align_corners=True)  # (B, reduced_channels, D, H, W)
            out = self.bn_relu(out)
            pyramids.append(out)
        return torch.cat(pyramids, dim=1)

# Squeeze-and-Excitation
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

# BNM
class BidomainNonlinearMapping(nn.Module):
    def __init__(self, nc, fusion_layers=2):
        super(BidomainNonlinearMapping, self).__init__()
        self.fusion_layers = fusion_layers
        self.process_real = nn.ModuleList([
            nn.Sequential(
                DepthwiseSeparableConv3D(nc, nc, kernel_size=1, padding=0),
                nn.ReLU(inplace=True),
                SELayer(channel=nc),
                DepthwiseSeparableConv3D(nc, nc, kernel_size=1, padding=0)
            ) for _ in range(fusion_layers)
        ])
        self.process_imag = nn.ModuleList([
            nn.Sequential(
                DepthwiseSeparableConv3D(nc, nc, kernel_size=1, padding=0),
                nn.ReLU(inplace=True),
                SELayer(channel=nc),
                DepthwiseSeparableConv3D(nc, nc, kernel_size=1, padding=0)
            ) for _ in range(fusion_layers)
        ])
        self.fusion_conv = nn.ModuleList([
            DepthwiseSeparableConv3D(nc * 3, nc, kernel_size=1, padding=0)
            for _ in range(fusion_layers)
        ])

    def forward(self, x):
        x_freq = torch.fft.fftn(x, dim=(2, 3, 4), norm='backward')
        for i in range(self.fusion_layers):
            real = x_freq.real
            imag = x_freq.imag
            real = self.process_real[i](real)
            imag = self.process_imag[i](imag)
            x_freq = torch.complex(real, imag)
            x_freq_spatial = torch.fft.ifftn(x_freq, dim=(2, 3, 4), norm='backward')

            x_freq_spatial_real = x_freq_spatial.real
            x_freq_spatial_imag = x_freq_spatial.imag

            x_freq_spatial_combined = torch.cat([x_freq_spatial_real, x_freq_spatial_imag], dim=1)
            x_cat = torch.cat([x, x_freq_spatial_combined], dim=1)
            x = self.fusion_conv[i](x_cat) + x
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.1, is_3d=False):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        if is_3d:
            self.fc1 = nn.Conv3d(in_features, hidden_features, kernel_size=1)
            self.dwconv = DepthwiseSeparableConv3D(hidden_features, hidden_features)
            self.act = act_layer(inplace=True)
            self.fc2 = nn.Conv3d(hidden_features, out_features, kernel_size=1)
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.dwconv = nn.Identity()
            self.act = act_layer(inplace=True)
            self.fc2 = nn.Linear(hidden_features, out_features)

        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SAS(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            drop_path: float = 0.1,
            norm_layer: nn.Module = nn.InstanceNorm3d,
            attn_drop_rate: float = 0.1,
            d_state: int = 16,
            dt_init: str = "random",
            mlp_ratio: float = 4.0,
            mlp_act_layer=nn.ReLU,
            mlp_drop_rate: float = 0.1,
            num_heads: int = 4,
    ):
        super().__init__()

        self.cpe1 = DepthwiseSeparableConv3D(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = WindowedSelfAttention3D(dim=hidden_dim, num_heads=num_heads, window_size=4,
                                                      dropout=attn_drop_rate)
        self.drop_path = nn.Identity()

        self.cpe2 = DepthwiseSeparableConv3D(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLP(
            in_features=hidden_dim,
            hidden_features=int(hidden_dim * mlp_ratio),
            act_layer=mlp_act_layer,
            drop=mlp_drop_rate,
            is_3d=True,
        )

        self.bidomain_mapping = BidomainNonlinearMapping(hidden_dim)

    def forward(self, x: torch.Tensor):

        x = x + self.cpe1(x)
        x_attn = checkpoint.checkpoint(self.self_attention, self.ln_1(x), use_reentrant=False)
        x = x + self.drop_path(x_attn)
        x = x + self.cpe2(x)
        # MLP
        x_mlp = checkpoint.checkpoint(self.mlp, self.ln_2(x), use_reentrant=False)
        x = x + self.drop_path(x_mlp)
        x_bidomain = self.bidomain_mapping(x)
        x = x + x_bidomain
        return x

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm_1', nn.InstanceNorm3d(num_input_features, affine=True))
        self.add_module('relu_1', nn.ReLU(inplace=True))
        self.add_module('conv_1',
                        nn.Conv3d(
                            num_input_features,
                            bn_size * growth_rate,
                            kernel_size=1,
                            stride=1,
                            bias=False))
        self.add_module('norm_2', nn.InstanceNorm3d(bn_size * growth_rate, affine=True))
        self.add_module('relu_2', nn.ReLU(inplace=True))
        self.add_module('conv_2',
                        nn.Conv3d(
                            bn_size * growth_rate,
                            growth_rate,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False))
        self.drop_rate = drop_rate
        self.dropout = nn.Dropout3d(self.drop_rate)

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = self.dropout(new_features)
        return torch.cat([x, new_features], 1)

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.InstanceNorm3d(num_input_features, affine=True))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv',
                        nn.Conv3d(
                            num_input_features,
                            num_output_features,
                            kernel_size=1,
                            stride=1,
                            bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))

class AdaptiveFusion(nn.Module):
    def __init__(self, in_channels):
        super(AdaptiveFusion, self).__init__()
        self.weight = nn.Parameter(torch.zeros(1, in_channels, 1, 1, 1))
        self.sigmoid = nn.Sigmoid()
        self.residual = DepthwiseSeparableConv3D(in_channels, in_channels, kernel_size=1, padding=0, bias=False)

    def forward(self, x1, x2):
        fusion_weight = self.sigmoid(self.weight)
        out = fusion_weight * x1 + (1 - fusion_weight) * x2
        out = self.residual(out) + (fusion_weight * x1 + (1 - fusion_weight) * x2)  # 残差连接
        return out

class ChannelAlign(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelAlign, self).__init__()
        self.conv1x1 = DepthwiseSeparableConv3D(in_channels, out_channels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        return self.conv1x1(x)

class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, pool_sizes=(1, 2, 3, 6)):
        super(PyramidPoolingModule, self).__init__()
        self.pool_sizes = pool_sizes
        self.pyramid_pooling = PyramidPooling3D(in_channels, pool_sizes=pool_sizes)

        reduced_channels = in_channels // len(pool_sizes)
        self.conv = DepthwiseSeparableConv3D(in_channels + len(pool_sizes) * reduced_channels, in_channels,
                                             kernel_size=1, padding=0)
        self.bn = nn.InstanceNorm3d(in_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pyramid_pooling(x)  # (B, in_channels + len(pool_sizes)*reduced_channels, D, H, W)
        x = self.conv(x)             # (B, in_channels, D, H, W)
        x = self.bn(x)
        x = self.relu(x)
        return x

class PT_DCN(nn.Module):
    def __init__(self):
        super(PT_DCN, self).__init__()
        growth_rate = 12
        num_layers = 2
        num_init_features = 8
        bn_size = 4
        drop_rate = 0.1
        num_classes = 2

        # 初始卷积
        self.features0 = nn.Sequential(
            OrderedDict([
                ('conv0',
                 DepthwiseSeparableConv3D(
                     1,
                     num_init_features,
                     kernel_size=3,
                     padding=1,
                     bias=False)),
                ('norm0', nn.InstanceNorm3d(num_init_features, affine=True)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
            ]))

        self.attention1 = SAS(
            hidden_dim=num_init_features * 2,
            drop_path=0.1,
            norm_layer=lambda num_features: nn.InstanceNorm3d(num_features, affine=True),
            attn_drop_rate=drop_rate,
            d_state=16,
            dt_init="random",
            mlp_ratio=4.0,
            mlp_act_layer=nn.ReLU,
            mlp_drop_rate=drop_rate,
            num_heads=4,
        )

        self.transferblock1 = nn.Sequential(
            OrderedDict([
                ('conv0',
                 DepthwiseSeparableConv3D(
                     num_init_features * 2,
                     num_init_features * 2,
                     kernel_size=1,
                     padding=0,
                     bias=False)),
                ('norm0', nn.InstanceNorm3d(num_init_features * 2, affine=True)),
                ('relu0', nn.ReLU(inplace=True)),
                ('conv1',
                 DepthwiseSeparableConv3D(
                     num_init_features * 2,
                     num_init_features,
                     kernel_size=3,
                     padding=1,
                     bias=False)),
            ]))

        self.adaptive_fusion1 = AdaptiveFusion(num_init_features)
        self.align1 = ChannelAlign(num_init_features, num_init_features)

        num_features = num_init_features * 2
        self.denseblock1_mri = _DenseBlock(
            num_layers=num_layers,
            num_input_features=num_features,
            bn_size=bn_size,
            growth_rate=growth_rate,
            drop_rate=drop_rate)
        self.denseblock1_pet = _DenseBlock(
            num_layers=num_layers,
            num_input_features=num_features,
            bn_size=bn_size,
            growth_rate=growth_rate,
            drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate

        self.transition1_mri = _Transition(
            num_input_features=num_features,
            num_output_features=num_features // 2)  # 16 -> 8
        self.transition1_pet = _Transition(
            num_input_features=num_features,
            num_output_features=num_features // 2)  # 16 -> 8
        num_features = num_features // 2
        self.pyramid_pooling1_mri = PyramidPoolingModule(num_features)
        self.pyramid_pooling1_pet = PyramidPoolingModule(num_features)

        self.attention3 = SAS(
            hidden_dim=num_features * 2,
            drop_path=0.1,
            norm_layer=lambda num_features: nn.InstanceNorm3d(num_features, affine=True),
            attn_drop_rate=drop_rate,
            d_state=16,
            dt_init="random",
            mlp_ratio=4.0,
            mlp_act_layer=nn.ReLU,
            mlp_drop_rate=drop_rate,
            num_heads=4,
        )

        self.transferblock3 = nn.Sequential(
            OrderedDict([
                ('conv0',
                 DepthwiseSeparableConv3D(
                     num_features * 2,
                     num_features * 2,
                     kernel_size=1,
                     padding=0,
                     bias=False)),
                ('norm0', nn.InstanceNorm3d(num_features * 2, affine=True)),
                ('relu0', nn.ReLU(inplace=True)),
                ('conv1',
                 DepthwiseSeparableConv3D(
                     num_features * 2,
                     num_features,
                     kernel_size=3,
                     padding=1,
                     bias=False)),
            ]))

        self.adaptive_fusion3 = AdaptiveFusion(num_features)
        self.align3 = ChannelAlign(num_features, num_features)

        num_features = num_features * 2
        self.denseblock3_mri = _DenseBlock(
            num_layers=num_layers,
            num_input_features=num_features,
            bn_size=bn_size,
            growth_rate=growth_rate,
            drop_rate=drop_rate)
        self.denseblock3_pet = _DenseBlock(
            num_layers=num_layers,
            num_input_features=num_features,
            bn_size=bn_size,
            growth_rate=growth_rate,
            drop_rate=drop_rate)
        num_features = num_features + num_layers * growth_rate

        self.transition3_mri = _Transition(
            num_input_features=num_features,
            num_output_features=num_features // 2)
        self.transition3_pet = _Transition(
            num_input_features=num_features,
            num_output_features=num_features // 2)
        num_features = num_features // 2

        self.pyramid_pooling3_mri = PyramidPoolingModule(num_features)
        self.pyramid_pooling3_pet = PyramidPoolingModule(num_features)


        self.features_end = nn.Sequential(
            OrderedDict([
                ('conv0',
                 DepthwiseSeparableConv3D(num_features * 2, num_features, kernel_size=3, padding=1, bias=False)),
                ('norm1', nn.InstanceNorm3d(num_features, affine=True)),
                ('relu1', nn.ReLU(inplace=True)),
                ('pool1', nn.AdaptiveAvgPool3d(1)),
            ]))

        self.classifier = nn.Linear(num_features, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm3d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, SELayer):
                for layer in m.fc:
                    if isinstance(layer, nn.Linear):
                        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                        if layer.bias is not None:
                            nn.init.constant_(layer.bias, 0)
            elif isinstance(m, DepthwiseSeparableConv3D):
                nn.init.kaiming_normal_(m.depthwise.weight, mode='fan_out', nonlinearity='relu')
                nn.init.kaiming_normal_(m.pointwise.weight, mode='fan_out', nonlinearity='relu')
                if m.depthwise.bias is not None:
                    nn.init.constant_(m.depthwise.bias, 0)
                if m.pointwise.bias is not None:
                    nn.init.constant_(m.pointwise.bias, 0)

    def forward(self, x, y):
        if x.dim() == 4:
            x = x.unsqueeze(1)
        if y.dim() == 4:
            y = y.unsqueeze(1)

        features_mri = self.features0(x)  # (B,4,D/2,H/2,W/2)
        features_pet = self.features0(y)  # (B,4,D/2,H/2,W/2)

        concat1 = torch.cat([features_mri, features_pet], 1)  # (B,8,D/2,H/2,W/2)
        concat1 = self.attention1(concat1)  # (B,8,D/2,H/2,W/2)
        output1 = self.transferblock1(concat1)  # (B,4,D/2,H/2,W/2)

        aligned1_mri = self.align1(features_mri)  # (B,4,D/2,H/2,W/2)
        aligned1_pet = self.align1(features_pet)  # (B,4,D/2,H/2,W/2)
        fused1 = self.adaptive_fusion1(aligned1_mri, aligned1_pet)  # (B,4,D/2,H/2,W/2)

        mri1 = torch.cat([fused1, output1], 1)  # (B,8,D/2,H/2,W/2)
        pet1 = torch.cat([fused1, output1], 1)  # (B,8,D/2,H/2,W/2)

        output1_mri = self.denseblock1_mri(mri1)  # (B,16, D/2, H/2, W/2)
        output1_pet = self.denseblock1_pet(pet1)  # (B,16, D/2, H/2, W/2)

        output1_mri = self.transition1_mri(output1_mri)  # (B,8, D/4, H/4, W/4)
        output1_pet = self.transition1_pet(output1_pet)  # (B,8, D/4, H/4, W/4)

        output1_mri = self.pyramid_pooling1_mri(output1_mri)  # (B,8,D/4,H/4,W/4)
        output1_pet = self.pyramid_pooling1_pet(output1_pet)  # (B,8,D/4,H/4,W/4)

        concat3 = torch.cat([output1_mri, output1_pet], 1)  # (B,16,D/4,H/4,W/4)
        concat3 = self.attention3(concat3)  # (B,16,D/4,H/4,W/4)
        output3 = self.transferblock3(concat3)  # (B,8,D/4,H/4,W/4)

        aligned3_mri = self.align3(output1_mri)  # (B,8,D/4,H/4,W/4)
        aligned3_pet = self.align3(output1_pet)  # (B,8,D/4,H/4,W/4)
        fused3 = self.adaptive_fusion3(aligned3_mri, aligned3_pet)  # (B,8,D/4,H/4,W/4)

        mri3 = torch.cat([fused3, output3], 1)  # (B,16,D/4,H/4,W/4)
        pet3 = torch.cat([fused3, output3], 1)  # (B,16,D/4,H/4,W/4)

        output3_mri = self.denseblock3_mri(mri3)  # (B,24, D/4, H/4, W/4)
        output3_pet = self.denseblock3_pet(pet3)  # (B,24, D/4, H/4, W/4)

        output3_mri = self.transition3_mri(output3_mri)  # (B,12, D/8, H/8, W/8)
        output3_pet = self.transition3_pet(output3_pet)  # (B,12, D/8, H/8, W/8)

        output3_mri = self.pyramid_pooling3_mri(output3_mri)  # (B,12,D/8,H/8,W/8)
        output3_pet = self.pyramid_pooling3_pet(output3_pet)  # (B,12,D/8,H/8,W/8)

        concat4 = torch.cat([output3_mri, output3_pet], 1)  # (B,24,D/8,H/8,W/8)
        output = self.features_end(concat4)  # (B,12,1,1,1)

        output_flat = output.view(output.size(0), -1)  # (B,12)
        output = self.classifier(output_flat)  # (B,2)
        return output
