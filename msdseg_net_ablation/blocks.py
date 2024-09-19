import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from zdence_mult_xiaorong.weight_init import trunc_normal_


class MaskedConv1D(nn.Module):
    """
    Masked 1D convolution. Interface remains the same as Conv1d.
    Only support a sub set of 1d convs
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode='zeros'
    ):
        super().__init__()
        # element must be aligned
        assert (kernel_size % 2 == 1) and (kernel_size // 2 == padding)
        # stride
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode)
        # zero out the bias term if it exists
        if bias:
            torch.nn.init.constant_(self.conv.bias, 0.)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()
        # input length must be divisible by stride
        assert T % self.stride == 0

        # conv
        out_conv = self.conv(x)
        # compute the mask
        if self.stride > 1:
            # downsample the mask using nearest neighbor
            out_mask = F.interpolate(
                mask.to(x.dtype),
                size=T // self.stride,
                mode='nearest'
            )
        else:
            # masking out the features
            out_mask = mask.to(x.dtype)

        # masking the output, stop grad to mask
        out_conv = out_conv * out_mask.detach()
        out_mask = out_mask.bool()
        return out_conv, out_mask


class LayerNorm(nn.Module):
    """
    LayerNorm that supports inputs of size B, C, T
    """

    def __init__(
            self,
            num_channels,
            eps=1e-5,
            affine=True,
            device=None,
            dtype=None,
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(
                torch.ones([1, num_channels, 1], **factory_kwargs))
            self.bias = nn.Parameter(
                torch.zeros([1, num_channels, 1], **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # assert x.dim() == 3
        # assert x.shape[1] == self.num_channels
        #  LayerNorm that supports inputs of size B, C, T
        # x = x.permute(0, 2, 3, 1)
        b, c, h, w = x.shape
        x = x.view(b, c, h * w)

        # normalization along C channels
        mu = torch.mean(x, dim=1, keepdim=True)
        res_x = x - mu
        sigma = torch.mean(res_x ** 2, dim=1, keepdim=True)
        out = res_x / torch.sqrt(sigma + self.eps)

        # apply weight and bias
        if self.affine:
            out *= self.weight
            out += self.bias
        out = out.view(b, c, h, w)
        return out


# helper functions for Transformer blocks
def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    # return a tensor of size 1 C T
    return torch.FloatTensor(sinusoid_table).unsqueeze(0).transpose(1, 2)


class ConvBlock(nn.Module):
    """
    A simple conv block similar to the basic block used in ResNet
    """

    def __init__(
            self,
            n_embd,  # dimension of the input features
            kernel_size=3,  # conv kernel size
            n_ds_stride=1,  # downsampling stride for the current layer
            expansion_factor=2,  # expansion factor of feat dims
            n_out=None,  # output dimension, if None, set to input dim
            act_layer=nn.ReLU,  # nonlinear activation used after conv, default ReLU
    ):
        super().__init__()
        # must use odd sized kernel
        assert (kernel_size % 2 == 1) and (kernel_size > 1)
        padding = kernel_size // 2
        if n_out is None:
            n_out = n_embd

        # 1x3 (strided) -> 1x3 (basic block in resnet)
        width = int(n_embd * expansion_factor)
        self.conv1 = MaskedConv1D(
            n_embd, width, kernel_size, n_ds_stride, padding=padding)
        self.conv2 = MaskedConv1D(
            width, n_out, kernel_size, 1, padding=padding)

        # attach downsampling conv op
        if n_ds_stride > 1:
            # 1x1 strided conv (same as resnet)
            self.downsample = MaskedConv1D(n_embd, n_out, 1, n_ds_stride)
        else:
            self.downsample = None

        self.act = act_layer()

    def forward(self, x, mask):
        identity = x
        out, out_mask = self.conv1(x, mask)
        out = self.act(out)
        out, out_mask = self.conv2(out, out_mask)

        # downsampling
        if self.downsample is not None:
            identity, _ = self.downsample(x, mask)

        # residual connection
        out += identity
        out = self.act(out)

        return out, out_mask


from mmcv.cnn import build_norm_layer


class Conv2d_BN(nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        self.inp_channel = a
        self.out_channel = b
        self.ks = ks
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.groups = groups

        self.add_module('c', nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        bn = build_norm_layer(norm_cfg, b)[1]
        nn.init.constant_(bn.weight, bn_weight_init)
        nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)


def get_shape(tensor):
    shape = tensor.shape
    if torch.onnx.is_in_onnx_export():
        shape = [i.cpu().numpy() for i in shape]
    return shape


class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads,
                 attn_ratio=4,
                 activation=None,
                 norm_cfg=dict(type='BN', requires_grad=True), ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        self.to_q = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_k = Conv2d_BN(dim, nh_kd, 1, norm_cfg=norm_cfg)
        self.to_v = Conv2d_BN(dim, self.dh, 1, norm_cfg=norm_cfg)

        self.proj = torch.nn.Sequential(activation(), Conv2d_BN(
            self.dh, dim, bn_weight_init=0, norm_cfg=norm_cfg))

    def forward(self, x):  # x (B,N,C)
        B, C, H, W = get_shape(x)

        qq = self.to_q(x).reshape(B, self.num_heads, self.key_dim, H * W).permute(0, 1, 3, 2)
        kk = self.to_k(x).reshape(B, self.num_heads, self.key_dim, H * W)
        vv = self.to_v(x).reshape(B, self.num_heads, self.d, H * W).permute(0, 1, 3, 2)

        attn = torch.matmul(qq, kk)
        attn = attn.softmax(dim=-1)  # dim = k

        xx = torch.matmul(attn, vv)

        xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, H, W)
        xx = self.proj(xx)
        return xx


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv2d_BN(in_features, hidden_features, norm_cfg=norm_cfg)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = Conv2d_BN(hidden_features, out_features, norm_cfg=norm_cfg)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SGPBlock(nn.Module):
    """
    A simple conv block similar to the basic block used in ResNet
    """

    def __init__(
            self,
            n_embd,  # dimension of the input features
            kernel_size=3,  # conv kernel size
            n_ds_stride=1,  # downsampling stride for the current layer
            k=1.5,  # k
            group=1,  # group for cnn
            n_out=None,  # output dimension, if None, set to input dim
            n_hidden=None,  # hidden dim for mlp
            path_pdrop=0.0,  # drop path rate
            act_layer=nn.GELU,  # nonlinear activation used after conv, default ReLU,
            downsample_type='max',
            init_conv_vars=1,  # init gaussian variance for the weight
            dim=256,
            key_dim=16,
            num_heads=8,
            mlp_ratio=2,
            attn_ratios=2,
            norm_cfg=dict(type='BN', requires_grad=True),
    ):
        super().__init__()
        # must use odd sized kernel
        # assert (kernel_size % 2 == 1) and (kernel_size > 1)
        # padding = kernel_size // 2

        self.kernel_size = kernel_size
        self.stride = n_ds_stride
        self.dim = n_embd
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        if n_out is None:
            n_out = n_embd

        self.ln = LayerNorm(n_embd)

        self.gn = nn.GroupNorm(16, n_embd)

        assert kernel_size % 2 == 1
        # add 1 to avoid have the same size as the instant-level branch
        up_size = round((kernel_size + 1) * k)
        up_size = up_size + 1 if up_size % 2 == 0 else up_size

        self.psi = nn.Conv2d(n_embd, n_embd, kernel_size, stride=1, padding=kernel_size // 2, groups=n_embd)
        self.fc1 = nn.Conv2d(n_embd, n_embd, 3, stride=1, padding=1, groups=n_embd)
        self.fc2 = nn.Conv2d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)
        self.convw = nn.Conv1d(n_embd, n_embd, kernel_size, stride=1, padding=kernel_size // 2, groups=n_embd)
        self.convkw = nn.Conv1d(n_embd, n_embd, up_size, stride=1, padding=up_size // 2, groups=n_embd)
        self.global_fc = nn.Conv2d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)

        # input
        if n_ds_stride > 1:
            if downsample_type == 'max':
                kernel_size, stride, padding = \
                    n_ds_stride + 1, n_ds_stride, (n_ds_stride + 1) // 2
                self.downsample = nn.MaxPool1d(
                    kernel_size, stride=stride, padding=padding)
                self.stride = stride
            elif downsample_type == 'avg':
                self.downsample = nn.Sequential(nn.AvgPool1d(n_ds_stride, stride=n_ds_stride, padding=0),
                                                nn.Conv1d(n_embd, n_embd, 1, 1, 0))
                self.stride = n_ds_stride
            else:
                raise NotImplementedError("downsample type error")
        else:
            self.downsample = nn.Identity()
            self.stride = 1

        # two layer mlp
        if n_hidden is None:
            n_hidden = 4 * n_embd  # default
        if n_out is None:
            n_out = n_embd

        mlp_hidden_dim = int(n_embd * mlp_ratio)
        self.attn = Attention(n_embd, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratios,
                              activation=act_layer,
                              norm_cfg=norm_cfg)
        self.mlp = Mlp(in_features=n_out, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=0.,
                       norm_cfg=norm_cfg)
        # drop path
        if path_pdrop > 0.0:
            self.drop_path_out = AffineDropPath(n_embd, drop_prob=path_pdrop)
            self.drop_path_mlp = AffineDropPath(n_out, drop_prob=path_pdrop)
        else:
            self.drop_path_out = nn.Identity()
            self.drop_path_mlp = nn.Identity()

        self.act = act_layer()
        self.reset_params(init_conv_vars=init_conv_vars)

    def reset_params(self, init_conv_vars=0):
        torch.nn.init.normal_(self.psi.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.fc1.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.fc2.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convw.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convkw.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.global_fc.weight, 0, init_conv_vars)
        torch.nn.init.constant_(self.psi.bias, 0)
        torch.nn.init.constant_(self.fc1.bias, 0)
        torch.nn.init.constant_(self.fc2.bias, 0)
        torch.nn.init.constant_(self.convw.bias, 0)
        torch.nn.init.constant_(self.convkw.bias, 0)
        torch.nn.init.constant_(self.global_fc.bias, 0)

    def forward(self, x):
        out = self.ln(x)  # torch.Size([1, 256, 16, 32])
        psi = self.psi(out)
        fc = self.fc1(out) + self.fc2(out)
        phi = torch.relu(self.global_fc(out.mean(dim=-1, keepdim=True)))
        out = fc * phi + (self.attn(out)) * psi + out

        out = x + self.drop_path_out(out)
        # FFN
        out = out + self.drop_path_mlp(self.mlp(self.ln(out)))

        return out


# class Block(nn.Module):
#
#     def __init__(self, dim, key_dim, num_heads, mlp_ratio=4., attn_ratio=2., drop=0.,
#                  drop_path=0., act_layer=nn.ReLU, norm_cfg=dict(type='BN2d', requires_grad=True)):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.mlp_ratio = mlp_ratio
#         self.ln = LayerNorm(dim)
#         self.kernel_size = kernel_size
#         self.stride = n_ds_stride
#         self.dim = n_embd
#         self.num_heads = num_heads
#         self.mlp_ratio = mlp_ratio
#         if n_out is None:
#             n_out = n_embd
#
#         self.ln = LayerNorm(n_embd)
#
#         self.gn = nn.GroupNorm(16, n_embd)
#
#         assert kernel_size % 2 == 1
#         # add 1 to avoid have the same size as the instant-level branch
#         up_size = round((kernel_size + 1) * k)
#         up_size = up_size + 1 if up_size % 2 == 0 else up_size
#
#         self.psi = nn.Conv1d(n_embd, n_embd, kernel_size, stride=1, padding=kernel_size // 2, groups=n_embd)
#         self.fc = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)
#         self.convw = nn.Conv1d(n_embd, n_embd, kernel_size, stride=1, padding=kernel_size // 2, groups=n_embd)
#         self.convkw = nn.Conv1d(n_embd, n_embd, up_size, stride=1, padding=up_size // 2, groups=n_embd)
#         self.global_fc = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)
#
#         self.gn = nn.GroupNorm(16, dim)
#         self.attn = Attention(dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio, activation=act_layer, norm_cfg=norm_cfg)
#
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, norm_cfg=norm_cfg)
#
#     def forward(self, x):
#
#         out = self.ln(x)  # torch.Size([1, 256, 16, 32])
#         print(out.shape)
#         psi = self.psi(out)
#         fc = self.fc(out)
#         # convw = self.convw(out)
#         # convkw = self.convkw(out)
#         phi = torch.relu(self.global_fc(out.mean(dim=-1, keepdim=True)))
#         out = fc * phi + (self.attn(out)) * psi + out
#
#         x = x + self.drop_path(out)
#         x = x + self.drop_path(self.mlp(self.gn(x)))
#         return x

def activation():
    return nn.ReLU(inplace=True)


def norm2d(out_channels):
    return nn.BatchNorm2d(out_channels)


class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                 bias=False, apply_act=True):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = norm2d(out_channels)
        if apply_act:
            self.act = activation()
        else:
            self.act = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class Trans_Decoder26(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        channels = {"8": 64, "16": 128, "32": 256}
        self.trans32 = SGPBlock(channels['32'])
        self.trans16 = SGPBlock(channels['16'])
        self.head32 = ConvBnAct(channels['32'], 128, 1)
        self.head16 = ConvBnAct(channels['16'], 128, 1)
        self.head8 = ConvBnAct(channels['8'], 8, 1)
        self.conv16 = ConvBnAct(128, 64, 3, 1, 1)
        self.conv8 = ConvBnAct(64 + 8, 64, 3, 1, 1)
        self.classifier = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x8, x16, x32 = x["8"], x["16"], x["32"]
        tx32 = self.trans32(x32)
        x32 = self.head32(tx32)
        tx16 = self.trans16(x16)
        x16 = self.head16(tx16)
        x8 = self.head8(x8)
        x32 = F.interpolate(x32, size=x16.shape[-2:], mode='bilinear', align_corners=False)
        x16 = x16 + x32
        x16 = self.conv16(x16)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x8 = torch.cat((x16, x8), dim=1)
        x8 = self.conv8(x8)
        x8 = self.classifier(x8)
        return x8


# drop path: from https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/common.py
class Scale(nn.Module):
    """
    Multiply the output regression range by a learnable constant value
    """

    def __init__(self, init_value=1.0):
        """
        init_value : initial value for the scalar
        """
        super().__init__()
        self.scale = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32),
            requires_grad=True
        )

    def forward(self, x):
        """
        input -> scale * input
        """
        return x * self.scale


# The follow code is modified from
# https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/common.py
def drop_path(x, drop_prob=0.0, training=False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
            x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class AffineDropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks) with a per channel scaling factor (and zero init)
    See: https://arxiv.org/pdf/2103.17239.pdf
    """

    def __init__(self, num_dim, drop_prob=0.0, init_scale_value=1e-4):
        super().__init__()
        self.scale = nn.Parameter(
            init_scale_value * torch.ones((1, num_dim, 1)),
            requires_grad=True
        )
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(self.scale * x, self.drop_prob, self.training)
