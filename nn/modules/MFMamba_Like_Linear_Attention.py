import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from timm.models.layers import DropPath
except ModuleNotFoundError:
    class DropPath(nn.Module):
        """Drop paths per sample, matching timm behavior for residual branches."""

        def __init__(self, drop_prob=0.0):
            super().__init__()
            self.drop_prob = drop_prob

        def forward(self, x):
            if self.drop_prob == 0.0 or not self.training:
                return x
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            return x.div(keep_prob) * random_tensor
try:
    from pytorch_wavelets import DWTForward, DWTInverse
except ModuleNotFoundError:
    DWTForward = DWTInverse = None
# https://arxiv.org/pdf/2503.11030


# 👉 MFM_Attention 做了这几件事：
#
# 输入特征
# → 位置编码（CPE）
# → 频域增强（FFT）
# → 多尺度卷积（3×3 / 5×5）
# → 线性注意力（Linear Attention + RoPE）
# → MLP
# → 残差融合
# → 输出


# 1️⃣ LinearAttention_B（线性注意力🔥） 👉 替代传统 Transformer Attention：
class LinearAttention_B(nn.Module):
    r""" Linear Attention with LePE and RoPE.
    """

    def __init__(self, dim, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.rope = RoPE()

    def forward(self, x):
        b, n, c = x.shape
        h = int(n ** 0.5)
        w = int(n ** 0.5)
        num_heads = self.num_heads
        head_dim = c // num_heads

        qk = self.qk(x).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        q, k, v = qk[0], qk[1], x

        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0
        q_rope = self.rope(q.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k_rope = self.rope(k.reshape(b, h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        q_rope = q_rope.to(q.dtype)
        k_rope = k_rope.to(k.dtype)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k_rope.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
        x = q_rope @ kv * z

        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x + self.lepe(v).permute(0, 2, 3, 1).reshape(b, n, c)

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'


# 2️⃣ RoPE（旋转位置编码） 👉 本质： 把位置编码变成“旋转”
class RoPE(torch.nn.Module):
    r"""Rotary Positional Embedding.
    """

    def __init__(self, base=10000):
        super(RoPE, self).__init__()
        self.base = base

    def forward(self, x):
        B, H, W, C = x.shape  # 假设输入是(B, H, W, C)格式

        # 确保特征维度是偶数，以便拆分为实部和虚部
        assert C % 2 == 0, f"特征维度必须是偶数，但得到{C}"

        # 计算旋转角度
        k_max = C // 2
        theta_ks = 1 / (self.base ** (torch.arange(k_max, device=x.device) / k_max))

        # 生成位置索引
        positions_h = torch.arange(H, device=x.device).unsqueeze(-1)
        positions_w = torch.arange(W, device=x.device).unsqueeze(-1)

        # 计算角度矩阵
        # 对于每个位置(h, w)，我们需要计算其对应的角度
        # 创建完整的位置矩阵
        pos_h = positions_h.repeat(1, W).unsqueeze(-1)  # [H, W, 1]
        pos_w = positions_w.repeat(H, 1).unsqueeze(-1).reshape(H, W, 1)  # [H, W, 1]

        # 每个位置(h, w)对应k_max个角度值
        # 我们将k_max分成两半，一半用于高度位置编码，一半用于宽度位置编码
        k_half = k_max // 2

        # 计算高度方向的角度
        angles_h = pos_h * theta_ks[:k_half]  # [H, W, k_half]

        # 计算宽度方向的角度
        angles_w = pos_w * theta_ks[k_half:]  # [H, W, k_half]

        # 组合角度矩阵
        angles = torch.cat([angles_h, angles_w], dim=-1)  # [H, W, k_max]

        # 生成旋转矩阵（复数表示）
        rotations = torch.polar(torch.ones_like(angles), angles)  # [H, W, k_max]

        # 准备输入张量用于复数乘法
        x_complex = torch.view_as_complex(x.reshape(B, H, W, k_max, 2))

        # 应用旋转位置编码
        x_rotated = x_complex * rotations

        # 转回实数表示并展平
        return torch.view_as_real(x_rotated).flatten(-2).to(x.dtype)


#
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# 三、MFM_Attention（核心🔥🔥🔥）

class MFM_Attention(nn.Module):
    def __init__(self, dim, out_channel, num_heads=4, mlp_ratio=4., qkv_bias=True,
                 drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.in_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.in_proj2 = nn.Conv2d(dim // 2, dim // 2, kernel_size=1)
        self.act_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.dwc = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.dwc2 = nn.Conv2d(dim // 2, dim // 2, 3, padding=1, groups=dim // 2)
        self.act = nn.SiLU()

        # 初始化注意力模块
        self.attn_s = LinearAttention_B(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.attn = LinearAttention_B(dim=dim // 2, num_heads=num_heads, qkv_bias=qkv_bias)

        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop
        )
        self.project_out = nn.Conv2d(dim, out_channel, kernel_size=1, bias=False)

        self.norm = nn.BatchNorm2d(dim)
        self.weight = nn.Sequential(
            nn.Conv2d(dim, dim // 16, 1, bias=True),
            nn.BatchNorm2d(dim // 16),
            nn.ReLU(True),
            nn.Conv2d(dim // 16, dim, 1, bias=True),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(True)

        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
        )

        self.reduce = nn.Sequential(
            nn.Conv2d(out_channel * 2, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True)
        )
        self.dwconv_3 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1, groups=dim // 2, bias=False)
        )
        self.dwconv_5 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Conv2d(dim, dim // 2, kernel_size=5, stride=1, padding=2, groups=dim // 2, bias=False)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # 保存原始维度用于后续恢复
        original_H, original_W = H, W

        # 确保H和W相等且为偶数
        target_size = max(H, W)
        if target_size % 2 != 0:
            target_size += 1

        # 如果H不等于W或不是偶数，则进行填充
        # Step 0️⃣ 输入预处理（对齐尺寸）
        # pad → 变成偶数 + 正方形
        if H != target_size or W != target_size:
            pad_h = target_size - H
            pad_w = target_size - W
            # 均匀填充左右和上下
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top

            x = nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
            H, W = target_size, target_size


        L = H * W
        x_0 = self.conv1(x)

        # 特征扁平化与位置编码融合
        x_flatten = x.flatten(2).permute(0, 2, 1)
        cpe1_out = self.cpe1(x).flatten(2).permute(0, 2, 1)
        x = x_flatten + cpe1_out
        shortcut = x

        # 频域特征处理
        x_freq = x.reshape(B, H, W, C).permute(0, 3, 1, 2).float()
        tepx = torch.fft.fft2(x_freq)
        weight_dtype = next(self.weight.parameters()).dtype
        norm_dtype = self.norm.weight.dtype
        weight_tepx = self.weight(tepx.real.to(weight_dtype)).to(tepx.real.dtype) * tepx
        fmt = torch.fft.ifft2(weight_tepx)
        fmt = torch.abs(fmt).to(norm_dtype)
        fmt = self.relu(self.norm(fmt)).flatten(2).permute(0, 2, 1)

        # 多尺度特征提取
        x_s = self.norm1(x)
        x_s3 = self.dwconv_3(x_s.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        x_s5 = self.dwconv_5(x_s.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)

        # 激活函数处理（调整维度为NCHW）
        act_proj_in = x_s.reshape(B, H, W, C).permute(0, 3, 1, 2)
        act_proj_out = self.act_proj(act_proj_in)
        act_res = self.act(act_proj_out.permute(0, 2, 3, 1).view(B, L, C))

        # 调整输入维度为NCHW格式
        x_s3_reshaped = x_s3.reshape(B, H, W, C // 2).permute(0, 3, 1, 2)
        x_s3 = self.in_proj2(x_s3_reshaped)
        x_s3 = self.act(self.dwc2(x_s3)).permute(0, 2, 3, 1).view(B, L, C // 2)

        x_s5_reshaped = x_s5.reshape(B, H, W, C // 2).permute(0, 3, 1, 2)
        x_s5 = self.in_proj2(x_s5_reshaped)
        x_s5 = self.act(self.dwc2(x_s5)).permute(0, 2, 3, 1).view(B, L, C // 2)

        # 线性注意力机制
        x_s3 = self.attn(x_s3)
        x_s5 = self.attn(x_s5)
        x_s = torch.cat((x_s3, x_s5), 2)

        # 特征融合与投影
        x_s_reshaped = x_s * act_res
        x_s_reshaped = x_s_reshaped.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x_s_proj = self.out_proj(x_s_reshaped)
        x_s_proj = x_s_proj.permute(0, 2, 3, 1).view(B, L, C)

        # 残差连接与位置编码
        x = shortcut + self.drop_path(x) + fmt
        x_cpe2 = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = x + self.cpe2(x_cpe2).flatten(2).permute(0, 2, 1)

        # 频域特征再次处理
        x_freq = x.reshape(B, H, W, C).permute(0, 3, 1, 2).float()
        tepx = torch.fft.fft2(x_freq)
        weight_dtype = next(self.weight.parameters()).dtype
        norm_dtype = self.norm.weight.dtype
        weight_tepx = self.weight(tepx.real.to(weight_dtype)).to(tepx.real.dtype) * tepx
        fmt = torch.fft.ifft2(weight_tepx)
        fmt = torch.abs(fmt).to(norm_dtype)
        fmt = self.relu(self.norm(fmt)).flatten(2).permute(0, 2, 1)

        # MLP处理
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)
        x = x + self.drop_path(x_mlp) + fmt

        # 输出投影与特征融合
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = self.project_out(x)
        x = self.reduce(torch.cat((x_0, x), 1)) + x_0

        # 恢复到原始维度
        if H != original_H or W != original_W:
            # 计算裁剪量
            crop_h = H - original_H
            crop_w = W - original_W
            crop_top = crop_h // 2
            crop_bottom = crop_h - crop_top
            crop_left = crop_w // 2
            crop_right = crop_w - crop_left

            x = x[:, :, crop_top:H - crop_bottom, crop_left:W - crop_right]

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}, mlp_ratio={self.mlp_ratio}"


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x = x.to(self.conv.weight.device)  # 确保输入张量在卷积层所在设备
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        x = x.to(self.conv.weight.device)  # 确保输入张量在卷积层所在设备
        return self.act(self.conv(x))


class StripConvBlock(nn.Module):
    """Strip convolution block for long horizontal, vertical and local CAD structures."""

    def __init__(self, channels):
        super().__init__()
        self.conv_h = Conv(channels, channels, (1, 11), 1, g=channels)
        self.conv_v = Conv(channels, channels, (11, 1), 1, g=channels)
        self.conv_l = Conv(channels, channels, 3, 1, g=channels)
        self.fuse = Conv(3 * channels, channels, 1, 1)

    def forward(self, x):
        return self.fuse(torch.cat((self.conv_h(x), self.conv_v(x), self.conv_l(x)), 1))


class EdgeEmbedding(nn.Module):
    """Learnable multi-scale edge embedding used to guide CAD line features."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.edge_3 = Conv(channels, channels, 3, 1, g=channels)
        self.edge_5 = Conv(channels, channels, 5, 1, g=channels)
        self.edge_7 = Conv(channels, channels, 7, 1, g=channels)
        self.fuse = Conv(3 * channels, channels, 1, 1)

        hidden = max(8, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
        )

    def forward(self, x):
        edge = self.fuse(torch.cat((self.edge_3(x), self.edge_5(x), self.edge_7(x)), 1))
        return self.refine(edge) * self.se(edge)


class EdgeGuidedMFMAttention(nn.Module):
    """Inject edge embeddings into MFM attention and fuse edge-aware responses."""

    def __init__(self, channels):
        super().__init__()
        self.edge = EdgeEmbedding(channels)
        self.mfm = MFM_Attention(channels, channels)
        self.fuse = Conv(2 * channels, channels, 1, 1)

    def forward(self, x):
        edge = self.edge(x)
        y = self.mfm(x + edge)
        return self.fuse(torch.cat((y, edge), 1))


class DirectionAwarePSA(nn.Module):
    """Separate height and width attention for long CAD lines and geometry."""

    def __init__(self, channels):
        super().__init__()
        self.conv_h = nn.Conv2d(channels, channels, 1, 1, 0, groups=channels)
        self.conv_w = nn.Conv2d(channels, channels, 1, 1, 0, groups=channels)

    def forward(self, x):
        h_attn = self.conv_h(F.adaptive_avg_pool2d(x, (x.shape[2], 1))).sigmoid()
        w_attn = self.conv_w(F.adaptive_avg_pool2d(x, (1, x.shape[3]))).sigmoid()
        return x * h_attn * w_attn


class LocalRefinement(nn.Module):
    """Depthwise local refinement followed by channel mixing."""

    def __init__(self, channels):
        super().__init__()
        self.refine = nn.Sequential(
            Conv(channels, channels, 3, 1, g=channels),
            Conv(channels, channels, 1, 1),
        )

    def forward(self, x):
        return x + self.refine(x)


class CoordAttention(nn.Module):
    """Coordinate attention applied after branch fusion."""

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


# 1️⃣ PSABloc_MFM_Attention
# x = x + attn(x)
# x = x + ffn(x)
#
# 👉 标准 Transformer Block： Attention + FFN
class AxisAwareLineAttention(nn.Module):
    """Axis-aware line attention for CAD horizontal, vertical and corner/diagonal structures."""

    def __init__(self, channels, k=11):
        super().__init__()
        self.line_h = Conv(channels, channels, (1, k), 1, g=channels)
        self.line_v = Conv(channels, channels, (k, 1), 1, g=channels)
        self.corner = Conv(channels, channels, 3, 1, g=channels)
        self.fuse = Conv(3 * channels, channels, 1, 1)
        self.attn = nn.Sigmoid()

        laplacian = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
        self.register_buffer("laplacian_kernel", laplacian)

    def _laplacian(self, x):
        kernel = self.laplacian_kernel.to(dtype=x.dtype, device=x.device).repeat(x.shape[1], 1, 1, 1)
        return F.conv2d(x, kernel, padding=1, groups=x.shape[1])

    def forward(self, x):
        fh = self.line_h(x)
        fv = self.line_v(x)
        fc = self.corner(x) + self._laplacian(x)
        attn = self.attn(self.fuse(torch.cat((fh, fv, fc), 1)))
        return x * attn


class PSABloc_MFM_Attention(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.direction = DirectionAwarePSA(c)
        self.local = LocalRefinement(c)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        y = self.local(self.direction(x))
        x = x + y if self.add else y
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


# 2️⃣ C2PSA_MFM_Attention
#
# 👉 YOLO里的 CSP结构： split → 一部分走attention → concat
#
# 👉 类似： C2f / C3模块
class C2PSA_MFM_Attention(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.coord_attn = nn.Identity()

        self.m = nn.Sequential(*(PSABloc_MFM_Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        x = x.to(self.cv1.conv.weight.device)  # 确保输入张量在第一个卷积层所在设备
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(self.coord_attn(torch.cat((a, b), 1)))


def main():
    """模型测试主函数"""
    batch_size, channels, height, width = 2, 32, 27, 31
    out_channels = 32
    num_heads = 4

    # 生成随机输入
    x = torch.randn(batch_size, channels, height, width)
    print(f"输入张量形状: {x.shape}")

    # 初始化模型
    model = MFM_Attention(
        dim=channels,
        out_channel=out_channels,
        num_heads=num_heads,
        mlp_ratio=4,
        drop=0.1,
        drop_path=0.1
    )
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 前向传播
    output = model(x)
    print(f"输出张量形状: {output.shape}")


if __name__ == "__main__":
    main()
