import pywt
import pywt.data
import torch
from torch import nn
from functools import partial
import torch.nn.functional as F

from .conv import Conv
from .block import C2f, C3, Bottleneck


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


# Wavelet Transform Conv(WTConv2d)
class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
                                   groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
                                                   groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


class DSConvWithWT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DSConvWithWT, self).__init__()

        # 深度卷积：使用 WTConv2d 替换 3x3 卷积
        self.depthwise = WTConv2d(in_channels, in_channels, kernel_size=kernel_size)

        # 逐点卷积：使用 1x1 卷积
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class StripConvBlock(nn.Module):
    """Strip convolution block with long horizontal, vertical and local depthwise branches."""

    def __init__(self, c1, c2):
        super().__init__()
        self.dw_h = Conv(c1, c1, (1, 11), 1, g=c1)
        self.dw_v = Conv(c1, c1, (11, 1), 1, g=c1)
        self.dw_l = Conv(c1, c1, 3, 1, g=c1)
        self.fuse = Conv(3 * c1, c2, 1, 1)

    def forward(self, x):
        return self.fuse(torch.cat((self.dw_h(x), self.dw_v(x), self.dw_l(x)), 1))


class CoordAttention(nn.Module):
    """Coordinate attention for separate height and width context modeling."""

    def __init__(self, channels, reduction=32):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.conv1 = Conv(channels, hidden, 1, 1)
        self.conv_h = nn.Conv2d(hidden, channels, 1, 1, 0)
        self.conv_w = nn.Conv2d(hidden, channels, 1, 1, 0)

    def forward(self, x):
        _, _, h, w = x.shape
        x_h = F.adaptive_avg_pool2d(x, (h, 1))
        x_w = F.adaptive_avg_pool2d(x, (1, w)).permute(0, 1, 3, 2)
        y = self.conv1(torch.cat((x_h, x_w), 2))
        y_h, y_w = torch.split(y, [h, w], 2)
        y_w = y_w.permute(0, 1, 3, 2)
        return x * self.conv_h(y_h).sigmoid() * self.conv_w(y_w).sigmoid()


class GeometryGatedCoordAttention(nn.Module):
    """Lightweight geometry gate for CAD neck features."""

    def __init__(self, channels, reduction=32):
        super().__init__()
        self.geo_gate = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 1, 1, 0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.geo_gate(x)


class FourierTokenMixer(nn.Module):
    """Mix global low-frequency structure and high-frequency detail in the Fourier domain."""

    def __init__(self, channels, reduction=4, cutoff=0.25):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.reduce = Conv(channels, hidden, 1, 1)
        self.project = Conv(hidden, channels, 1, 1, act=False)
        self.alpha = nn.Parameter(torch.zeros(1, hidden, 1, 1))
        self.cutoff = cutoff

    def _frequency_masks(self, h, w, device):
        fy = torch.fft.fftfreq(h, device=device).view(h, 1)
        fx = torch.fft.fftfreq(w, device=device).view(1, w)
        radius = torch.sqrt(fy.square() + fx.square())
        low = torch.exp(-0.5 * (radius / self.cutoff).square()).view(1, 1, h, w)
        return low, 1.0 - low

    def forward(self, x):
        identity = x
        tokens = self.reduce(x)
        input_dtype = tokens.dtype

        # FFT is kept in FP32 for CUDA AMP stability.
        freq = torch.fft.fft2(tokens.float(), dim=(-2, -1), norm="ortho")
        low_mask, high_mask = self._frequency_masks(tokens.shape[-2], tokens.shape[-1], tokens.device)
        low_freq = freq * low_mask
        high_freq = freq * high_mask

        alpha = self.alpha.sigmoid().float()
        mixed_freq = alpha * low_freq + (1.0 - alpha) * high_freq
        mixed = torch.fft.ifft2(mixed_freq, dim=(-2, -1), norm="ortho").real.to(input_dtype)
        return identity + self.project(mixed)


class LearnableEdgeFusion(nn.Module):
    """Learn multi-scale edge responses and refine them with spatial and channel interaction."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.edge_3 = Conv(channels, channels, 3, 1, g=channels)
        self.edge_5 = Conv(channels, channels, 5, 1, g=channels)
        self.edge_7 = Conv(channels, channels, 7, 1, g=channels)

        self.channel_fuse = Conv(3 * channels, channels, 1, 1)
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )

        hidden = max(8, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.act = nn.GELU()

    def forward(self, x):
        edge = torch.cat((self.edge_3(x), self.edge_5(x), self.edge_7(x)), 1)
        edge = self.channel_fuse(edge)
        return self.act(self.spatial_refine(edge)) * self.se(edge)


class PartialWTBottleneck(nn.Module):
    """CAD-oriented bottleneck: local detail first, then partial wavelet modeling."""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        hidden = max(1, int(c2 * e))
        wt_channels = hidden // 2
        self.cv1 = Conv(c1, hidden, 3, 1)
        self.dw = Conv(hidden, hidden, 3, 1, g=hidden)
        self.wt_channels = wt_channels
        self.wt_b3 = WTConv2d(wt_channels, wt_channels) if wt_channels > 0 else nn.Identity()
        self.wt_b4 = WTConv2d(wt_channels, wt_channels) if wt_channels > 0 else nn.Identity()
        self.cv2 = Conv(hidden, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.dw(self.cv1(x))
        if self.wt_channels > 0:
            y_keep, y_wt = torch.split(y, [y.shape[1] - self.wt_channels, self.wt_channels], 1)
            y = torch.cat((y_keep, self.wt_b4(self.wt_b3(y_wt))), 1)
        y = self.cv2(y)
        return x + y if self.add else y


class Bottleneck_WT(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = WTConv2d(c_, c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3k_WT(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_WT(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# 在c3k=True时，使用Bottleneck_WT特征融合，为false的时候我们使用普通的Bottleneck提取特征
class C3k2_WT(C2f):
    """CAD-oriented C3k2_WT with ACConv, partial WT, strip attention and learnable edge fusion."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initialize the CAD-C3K2-WT module while keeping the original YAML signature."""
        super().__init__(c1, c2, n, shortcut, g, e)
        depth = n * (2 if c3k else 1)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.acconv = StripConvBlock(self.c, self.c)
        self.m = nn.Sequential(*(PartialWTBottleneck(self.c, self.c, shortcut, g, e=0.5) for _ in range(depth)))
        self.edge = LearnableEdgeFusion(self.c)
        self.attn = FourierTokenMixer(self.c)
        self.cv2 = Conv(2 * self.c, c2, 1, 1)

    def forward(self, x):
        """Forward pass matching the CAD-C3K2-WT diagram."""
        shortcut, main = self.cv1(x).chunk(2, 1)
        main = self.acconv(main)
        main = self.m(main)
        main = self.attn(main + self.edge(shortcut))
        return self.cv2(torch.cat((main, shortcut), 1))


class CADBiFPNAdd(nn.Module):
    """Weighted BiFPN-style feature fusion for same-resolution feature maps."""

    def __init__(self, n=2, eps=1e-4):
        super().__init__()
        self.w = nn.Parameter(torch.ones(n, dtype=torch.float32), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        w = F.relu(self.w)
        w = w / (w.sum() + self.eps)
        return sum(w[i] * x[i] for i in range(len(x)))


class GeometryAwareDetailFusion(nn.Module):
    """Geometry-aware high-resolution detail fusion for CAD P2 features."""

    def __init__(self, c1, c2):
        super().__init__()
        self.cv = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.geo_dw = Conv(c2, c2, 3, 1, g=c2)

        self.fuse = Conv(c2, c2, 1, 1)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(c2, c2, 3, 1, 1, groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(),
            nn.Conv2d(c2, c2, 1, 1, 0, bias=True),
            nn.Sigmoid(),
        )

        laplacian = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
        self.register_buffer("laplacian_kernel", laplacian)

    def _laplacian(self, x):
        kernel = self.laplacian_kernel.to(dtype=x.dtype, device=x.device).repeat(x.shape[1], 1, 1, 1)
        return F.conv2d(x, kernel, padding=1, groups=x.shape[1])

    def forward(self, x):
        x = self.cv(x)
        geo = self.geo_dw(x) + self._laplacian(x)
        fused = self.fuse(geo)
        return x + fused * self.spatial_gate(fused)


class DSConvDown(nn.Module):
    """Depthwise separable downsampling used in the bottom-up CAD neck path."""

    def __init__(self, c1, c2, k=3):
        super().__init__()
        self.dw = Conv(c1, c1, k, 2, g=c1)
        self.pw = Conv(c1, c2, 1, 1)

    def forward(self, x):
        return self.pw(self.dw(x))


class CADNeckBlock(nn.Module):
    """CAD-C3K2-WT neck enhancement."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__()
        self.block = C3k2_WT(c1, c2, n, c3k, e, g, shortcut)

    def forward(self, x):
        return self.block(x)


if __name__ == '__main__':
    DW = DSConvWithWT(256, 128)
    #创建一个输入张量
    batch_size = 8
    input_tensor=torch.randn(batch_size, 256, 64, 64 )
    #运行模型并打印输入和输出的形状
    output_tensor =DW(input_tensor)
    print("Input shape:",input_tensor.shape)
    print("0utput shape:",output_tensor.shape)
