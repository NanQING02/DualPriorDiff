import math
import torch
import numbers
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
from einops import rearrange


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
# 正余弦位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear")    # nearest
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class AttentionWithPrio(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(AttentionWithPrio, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.kernel = nn.Sequential(
            nn.Linear(256, dim * 2, bias=False),
        )
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, k_v):
        b, c, h, w = x.shape
        k_v = self.kernel(k_v).view(-1, c * 2, 1, 1)
        k_v1, k_v2 = k_v.chunk(2, dim=1)
        x = x * k_v1 + k_v2

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_BatchNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_BatchNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_BatchNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_BatchNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        # nn.Parameter()将一个不可训练的tensor转换成可以训练的类型parameter，
        # 并将这个parameter绑定到这个module里面。即在定义网络时这个tensor就是一个可以训练的参数了。
        # 使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        # 最后一个维度的均值
        mu = x.mean(-1, keepdim=True)
        # 最后一个维度的方差
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class BatchNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(BatchNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_BatchNorm(dim)
        else:
            self.body = WithBias_BatchNorm(dim)

    def forward(self, x):
        # 获取x的最后2个维度
        h, w = x.shape[-2:]
        # to_3d()将图片在空间维度 h 和 w 上展平，在c通道上做层归一化
        return to_4d(self.body(to_3d(x)), h, w)


# 先验融合模块
class Priori_Fusion_Block(nn.Module):
    def __init__(self, dim, hidden_factor, bias):
        super(Priori_Fusion_Block, self).__init__()

        hidden_features = int(dim*hidden_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.hidden_conv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.kernel = nn.Sequential(
            nn.Linear(256, dim*2, bias=False),
        )

    def forward(self, x, k_v):
        b, c, h, w = x.shape
        k_v = self.kernel(k_v).view(-1, c*2, 1, 1)
        k_v1, k_v2 = k_v.chunk(2, dim=1)
        x = x*k_v1+k_v2
        x = self.project_in(x)
        x1, x2 = self.hidden_conv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class PIAR(nn.Module):
    def __init__(self, dim, dim_out, num_heads, norm_groups, hidden_factor, bias, LayerNorm_type, noise_level_emb_dim=None, dropout=0):      # 'WithBias', Other option 'BiasFree'
        super(PIAR, self).__init__()

        # self.block = Block(dim, dim_out, groups=norm_groups)
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        self.norm1 = BatchNorm(dim_out, LayerNorm_type)
        self.attn = AttentionWithPrio(dim_out, num_heads, bias)
        self.norm2 = BatchNorm(dim_out, LayerNorm_type)
        self.pfb = Priori_Fusion_Block(dim_out, hidden_factor, bias)

    def forward(self, x, t, k_value):
        x = self.res_block(x, t)
        # x = self.block(x)
        x = x + self.attn(self.norm1(x), k_value)
        x = x + self.pfb(self.norm2(x), k_value)

        return x


class UNet(nn.Module):
    """Denoising U-Net with prior injection and configurable skip routing.

    Args:
        encoder_blocks: block type of every encoder level, one entry per
            ``channel_mults`` entry, e.g. ``"PIAR,PIAR,PIAR,PIAR,RAB,RAB"``.

            * ``PIAR`` - prior-infused attention residual block (consumes the prior vector)
            * ``RAB``  - residual block with self-attention
            * ``RB``   - plain residual block
        use_prior: when ``False`` every ``PIAR`` level falls back to ``RB`` and the
            prior vector is ignored, which gives the no-prior baseline.
        skip_mode: ``"dense"`` routes every encoder layer output to the mirror decoder
            block; ``"plain"`` gives the encoder stage output only to the first decoder
            block of the matching resolution (standard U-Net skip).
    """

    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=64,
        norm_groups=32,
        channel_mults=(1, 2, 4, 4, 8, 8),
        attn_res=(16, 8),
        res_blocks=3,
        dropout=0.,
        hidden_factor=2,
        with_noise_level_emb=True,
        bias=False,
        LayerNorm_type="WithBias",
        image_size=256,
        encoder_blocks=None,
        use_prior=True,
        skip_mode="dense",
    ):
        super().__init__()

        num_mults = len(channel_mults)
        if encoder_blocks is None:
            n_prior = max(num_mults - len(attn_res), 0)
            encoder_blocks = ["PIAR"] * n_prior + ["RAB"] * (num_mults - n_prior)
        if isinstance(encoder_blocks, str):
            encoder_blocks = [b.strip().upper() for b in encoder_blocks.split(",") if b.strip()]
        if len(encoder_blocks) != num_mults:
            raise ValueError(
                "encoder_blocks must have %d entries (one per channel_mults entry), got %d"
                % (num_mults, len(encoder_blocks)))
        for b in encoder_blocks:
            if b not in ("PIAR", "RAB", "RB"):
                raise ValueError("unknown encoder block type: %s" % b)
        if skip_mode not in ("dense", "plain"):
            raise ValueError("skip_mode must be 'dense' or 'plain'")

        self.use_prior = use_prior
        self.skip_mode = skip_mode
        self.encoder_blocks = encoder_blocks

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        # ---- encoder ----
        pre_channel = inner_channel
        now_res = image_size
        feat_meta = []          # (out_channels, resolution) of every encoder output
        downs = [nn.Conv2d(in_channel, inner_channel, kernel_size=3, padding=1)]
        feat_meta.append((inner_channel, now_res))

        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                block = encoder_blocks[ind]
                if block == "PIAR" and use_prior:
                    downs.append(PIAR(dim=pre_channel, dim_out=channel_mult, num_heads=channel_mults[ind],
                                      norm_groups=norm_groups, hidden_factor=hidden_factor,
                                      bias=bias, LayerNorm_type=LayerNorm_type,
                                      noise_level_emb_dim=noise_level_channel, dropout=dropout))
                else:
                    downs.append(ResnetBlocWithAttn(
                        dim=pre_channel, dim_out=channel_mult, noise_level_emb_dim=noise_level_channel,
                        norm_groups=norm_groups, dropout=dropout, with_attn=(use_attn or block == "RAB")))
                feat_meta.append((channel_mult, now_res))
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                now_res = now_res // 2
                feat_meta.append((pre_channel, now_res))

        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                               norm_groups=norm_groups, dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                               norm_groups=norm_groups, dropout=dropout, with_attn=False)
        ])

        # ---- decoder ----
        boundary_idx = {}       # last encoder output at each resolution
        for i, (_, res) in enumerate(feat_meta):
            boundary_idx[res] = i

        ups = []
        skip_plan = []          # encoder output index consumed by each decoder layer
        now_res = image_size >> (num_mults - 1)
        ptr = len(feat_meta) - 1

        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for k in range(0, res_blocks + 1):
                if skip_mode == "dense":
                    skip_ch = feat_meta[ptr][0]
                    skip_plan.append(ptr)
                    ptr -= 1
                elif k == 0:
                    skip_ch = feat_meta[boundary_idx[now_res]][0]
                    skip_plan.append(boundary_idx[now_res])
                else:
                    skip_ch = 0
                    skip_plan.append(None)
                ups.append(ResnetBlocWithAttn(
                    pre_channel + skip_ch, channel_mult, noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                skip_plan.append(None)
                now_res = now_res * 2

        if skip_mode == "dense" and ptr != -1:
            raise RuntimeError("dense skip bookkeeping is inconsistent (leftover encoder outputs)")

        self.ups = nn.ModuleList(ups)
        self.skip_plan = skip_plan
        self.kept_feats = sorted({i for i in skip_plan if i is not None})

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

    def forward(self, x, time, k_value=None):
        t = self.noise_level_mlp(time) if exists(self.noise_level_mlp) else None

        keep = set(self.kept_feats)
        feats = {}
        for i, layer in enumerate(self.downs):
            if isinstance(layer, PIAR):
                x = layer(x, t, k_value)
            elif isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            if i in keep:
                feats[i] = x

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)

        for layer, idx in zip(self.ups, self.skip_plan):
            if isinstance(layer, ResnetBlocWithAttn):
                if idx is None:
                    x = layer(x, t)
                else:
                    x = layer(torch.cat((x, feats[idx]), dim=1), t)
            else:
                x = layer(x)

        return self.final_conv(x)


def create_unet(
    encoder_blocks="PIAR,PIAR,PIAR,PIAR,RAB,RAB",
    use_prior=True,
    skip_mode="dense",
    image_size=256,
    in_channel=6,
    out_channel=3,
    inner_channel=64,
    channel_mults=(1, 2, 4, 4, 8, 8),
    attn_res=(16, 8),
    res_blocks=3,
    dropout=0.0,
    norm_groups=32,
    hidden_factor=2,
):
    """Build the denoising network used in the paper."""
    return UNet(
        in_channel=in_channel,
        out_channel=out_channel,
        inner_channel=inner_channel,
        norm_groups=norm_groups,
        channel_mults=channel_mults,
        attn_res=attn_res,
        res_blocks=res_blocks,
        dropout=dropout,
        hidden_factor=hidden_factor,
        image_size=image_size,
        encoder_blocks=encoder_blocks,
        use_prior=use_prior,
        skip_mode=skip_mode,
    )


def create_sr256_unet(use_prior=True, skip_mode="dense", encoder_blocks="PIAR,PIAR,PIAR,PIAR,RAB,RAB"):
    return create_unet(encoder_blocks=encoder_blocks, use_prior=use_prior,
                       skip_mode=skip_mode, image_size=256)


def create_sr384_unet(use_prior=True, skip_mode="dense", encoder_blocks="PIAR,PIAR,PIAR,PIAR,RAB,RAB"):
    return create_unet(encoder_blocks=encoder_blocks, use_prior=use_prior, skip_mode=skip_mode,
                       image_size=384, inner_channel=96, attn_res=(16,))


if __name__ == "__main__":
    cat_img = torch.Tensor(2, 6, 256, 256)
    t = torch.Tensor(2, 1)
    k_value = torch.Tensor(2, 256)
    for prior, skip in [(True, "dense"), (True, "plain"), (False, "dense")]:
        net = create_sr256_unet(use_prior=prior, skip_mode=skip)
        total = sum(p.nelement() for p in net.parameters())
        out = net(cat_img, t, k_value if prior else None)
        print("prior=%-5s skip=%-5s params=%.2fM out=%s" % (prior, skip, total / 1e6, tuple(out.shape)))
