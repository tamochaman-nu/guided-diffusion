# This file is adapted from NVlabs/edm2, training/networks_edm2.py
#   https://github.com/NVlabs/edm2
#   Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#   Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0
#   International License (CC BY-NC-SA 4.0): http://creativecommons.org/licenses/by-nc-sa/4.0/
# Accordingly, this file (and only this file) is distributed under CC BY-NC-SA 4.0.
#
# Changes from the original: adapted to guided-diffusion's interface (discrete
# timesteps, epsilon/learn_sigma outputs, integer class labels, guided-diffusion
# hyperparameters), fp32 normalization for half-precision activations, optional
# gradient checkpointing, scaled_dot_product_attention for the cosine attention,
# and box-filter resampling via avg_pool2d / nearest interpolation.

"""
Magnitude-preserving U-Net (EDM2, "Analyzing and Improving the Training Dynamics
of Diffusion Models", Karras et al. 2024, CONFIG G) for guided-diffusion.
"""

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .gaussian_diffusion import get_named_beta_schedule


def normalize(x, dim=None, eps=1e-4):
    """
    Scale x to unit RMS over `dim` (default: all but the first), computed in fp32
    (EDM2 Algorithm 1).
    """
    if dim is None:
        dim = list(range(1, x.ndim))
    xf = x.float()
    norm = th.linalg.vector_norm(xf, dim=dim, keepdim=True)
    norm = th.add(eps, norm, alpha=math.sqrt(norm.numel() / x.numel()))
    return (xf / norm).to(x.dtype)


def mp_silu(x):
    """Magnitude-preserving SiLU (EDM2 Eq. 81)."""
    return F.silu(x) / 0.596


def mp_sum(a, b, t=0.5):
    """Magnitude-preserving sum (EDM2 Eq. 88)."""
    return a.lerp(b, t) / math.sqrt((1 - t) ** 2 + t ** 2)


def mp_cat(a, b, dim=1, t=0.5):
    """Magnitude-preserving concatenation (EDM2 Eq. 103)."""
    na, nb = a.shape[dim], b.shape[dim]
    c = math.sqrt((na + nb) / ((1 - t) ** 2 + t ** 2))
    wa = c / math.sqrt(na) * (1 - t)
    wb = c / math.sqrt(nb) * t
    return th.cat([wa * a, wb * b], dim=dim)


class MPFourier(nn.Module):
    """Magnitude-preserving Fourier features (EDM2 Eq. 75), fixed (non-trained) buffers."""

    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer("freqs", 2 * np.pi * th.randn(num_channels) * bandwidth)
        self.register_buffer("phases", 2 * np.pi * th.rand(num_channels))

    def forward(self, x):
        y = x.float().ger(self.freqs.float())
        y = y + self.phases.float()
        return y.cos() * math.sqrt(2)


class MPConv(nn.Module):
    """
    Magnitude-preserving convolution / fully-connected layer (EDM2 Eq. 47) with
    forced weight normalization (Eq. 66). No bias; weights are kept in fp32 and
    cast to the activation dtype only when used.
    """

    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.out_channels = out_channels
        self.weight = nn.Parameter(th.randn(out_channels, in_channels, *kernel))

    def forward(self, x, gain=1):
        w = self.weight.float()
        if self.training:
            with th.no_grad():
                self.weight.copy_(normalize(w))  # forced weight normalization
        w = normalize(w)  # traditional weight normalization
        w = w * (gain / math.sqrt(w[0].numel()))  # magnitude-preserving scaling
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 4
        return F.conv2d(x, w, padding=(w.shape[-1] // 2,))


def resample(x, mode="keep"):
    """Box-filter resampling on the main path (EDM2 resample() with f=[1,1])."""
    if mode == "keep":
        return x
    if mode == "down":
        return F.avg_pool2d(x, 2)
    assert mode == "up"
    return F.interpolate(x, scale_factor=2, mode="nearest")


class MPBlock(nn.Module):
    """U-Net encoder/decoder block with optional cosine self-attention (EDM2 Fig. 21)."""

    def __init__(
        self,
        in_channels,
        out_channels,
        emb_channels,
        flavor="enc",
        resample_mode="keep",
        attention=False,
        num_heads=1,
        dropout=0.0,
        res_balance=0.3,
        attn_balance=0.3,
        clip_act=256,
        use_checkpoint=False,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.flavor = flavor
        self.resample_mode = resample_mode
        self.num_heads = num_heads if attention else 0
        self.dropout = dropout
        self.res_balance = res_balance
        self.attn_balance = attn_balance
        self.clip_act = clip_act
        self.use_checkpoint = use_checkpoint
        self.emb_gain = nn.Parameter(th.zeros([]))
        self.conv_res0 = MPConv(
            out_channels if flavor == "enc" else in_channels, out_channels, kernel=[3, 3]
        )
        self.emb_linear = MPConv(emb_channels, out_channels, kernel=[])
        self.conv_res1 = MPConv(out_channels, out_channels, kernel=[3, 3])
        self.conv_skip = (
            MPConv(in_channels, out_channels, kernel=[1, 1])
            if in_channels != out_channels
            else None
        )
        if self.num_heads:
            assert out_channels % self.num_heads == 0
            self.attn_qkv = MPConv(out_channels, out_channels * 3, kernel=[1, 1])
            self.attn_proj = MPConv(out_channels, out_channels, kernel=[1, 1])
        else:
            self.attn_qkv = self.attn_proj = None

    def forward(self, x, emb):
        if self.use_checkpoint and self.training and th.is_grad_enabled():
            return checkpoint(self._forward, x, emb, use_reentrant=False)
        return self._forward(x, emb)

    def _forward(self, x, emb):
        # Main branch.
        x = resample(x, mode=self.resample_mode)
        if self.flavor == "enc":
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = normalize(x, dim=1)  # pixel norm

        # Residual branch: conditioning is multiplicative only (no shift).
        y = self.conv_res0(mp_silu(x))
        c = self.emb_linear(emb, gain=self.emb_gain) + 1
        y = mp_silu(y * c[:, :, None, None].to(y.dtype))
        if self.training and self.dropout != 0:
            y = F.dropout(y, p=self.dropout)
        y = self.conv_res1(y)

        # Connect the branches.
        if self.flavor == "dec" and self.conv_skip is not None:
            x = self.conv_skip(x)
        x = mp_sum(x, y, t=self.res_balance)

        # Cosine self-attention: q, k, v are pixel-normalized per head, so the
        # logits are bounded and fp16-safe. SDPA computes the same softmax(qk^T/sqrt(c))v.
        if self.num_heads:
            n, ch, h, w = x.shape
            y = self.attn_qkv(x)
            y = y.reshape(n, self.num_heads, -1, 3, h * w)
            q, k, v = normalize(y, dim=2).unbind(3)  # each [n, heads, c, hw]
            y = F.scaled_dot_product_attention(
                q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2)
            ).transpose(-1, -2)
            y = self.attn_proj(y.reshape(n, ch, h, w))
            x = mp_sum(x, y, t=self.attn_balance)

        if self.clip_act is not None:
            x = x.clamp(-self.clip_act, self.clip_act)
        return x


class MPUNetModel(nn.Module):
    """
    EDM2 magnitude-preserving U-Net with guided-diffusion's interface:
    forward(x, timesteps, y=None) -> [N x out_channels x H x W].

    :param attention_resolutions: downsample rates with attention (as in UNetModel).
    :param num_head_channels: channels per head; -1 means use num_heads heads.
    :param diffusion_steps, noise_schedule: used to map the discrete timestep to
        the noise level fed to the Fourier embedding (c_noise = log(sigma) / 4).
    :param clip_act: clip block outputs to [-clip_act, clip_act]; None disables.
    """

    # Weights stay fp32 even in fp16/bf16 mode (only activations are half), so the
    # parameters themselves are the master weights and forced weight normalization
    # acts on the weights the optimizer updates. See MixedPrecisionTrainer.
    fp32_master_weights = True

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0.0,
        channel_mult=(1, 2, 4, 8),
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        use_bf16=False,
        num_heads=1,
        num_head_channels=-1,
        diffusion_steps=1000,
        noise_schedule="linear",
        clip_act=256,
        res_balance=0.3,
        attn_balance=0.3,
        label_balance=0.5,
        concat_balance=0.5,
    ):
        super().__init__()
        self.image_size = image_size
        self.out_channels = out_channels
        self.num_classes = num_classes
        self.dtype = (
            th.bfloat16 if use_bf16 else th.float16 if use_fp16 else th.float32
        )
        self.label_balance = label_balance
        self.concat_balance = concat_balance
        self.out_gain = nn.Parameter(th.zeros([]))

        cblock = [int(model_channels * m) for m in channel_mult]
        cnoise = cblock[0]
        cemb = max(cblock)
        attn_res = {image_size // ds for ds in attention_resolutions}

        def heads(ch):
            if num_head_channels == -1:
                return num_heads
            assert ch % num_head_channels == 0
            return ch // num_head_channels

        def block(cin, cout, attention=False, **kw):
            return MPBlock(
                cin,
                cout,
                cemb,
                attention=attention,
                num_heads=heads(cout) if attention else 0,
                dropout=dropout,
                res_balance=res_balance,
                attn_balance=attn_balance,
                clip_act=clip_act,
                use_checkpoint=use_checkpoint,
                **kw,
            )

        # Noise level per discrete timestep: c_noise = log(sigma_t) / 4 with
        # sigma_t = sqrt((1 - alphabar_t) / alphabar_t) (EDM2 uses log(sigma) / 4).
        alphas_cumprod = np.cumprod(1.0 - get_named_beta_schedule(noise_schedule, diffusion_steps))
        log_sigma = 0.5 * np.log((1.0 - alphas_cumprod) / alphas_cumprod)
        self.register_buffer(
            "c_noise_table", th.tensor(log_sigma / 4, dtype=th.float32), persistent=False
        )

        # Embedding.
        self.emb_fourier = MPFourier(cnoise)
        self.emb_noise = MPConv(cnoise, cemb, kernel=[])
        self.emb_label = MPConv(num_classes, cemb, kernel=[]) if num_classes else None

        # Encoder.
        self.enc = nn.ModuleDict()
        cout = in_channels + 1  # +1: constant-one channel instead of biases
        for level, channels in enumerate(cblock):
            res = image_size >> level
            if level == 0:
                cin, cout = cout, channels
                self.enc[f"{res}x{res}_conv"] = MPConv(cin, cout, kernel=[3, 3])
            else:
                self.enc[f"{res}x{res}_down"] = block(cout, cout, flavor="enc", resample_mode="down")
            for idx in range(num_res_blocks):
                cin, cout = cout, channels
                self.enc[f"{res}x{res}_block{idx}"] = block(
                    cin, cout, flavor="enc", attention=res in attn_res
                )

        # Decoder.
        self.dec = nn.ModuleDict()
        skips = [b.out_channels for b in self.enc.values()]
        for level, channels in reversed(list(enumerate(cblock))):
            res = image_size >> level
            if level == len(cblock) - 1:
                self.dec[f"{res}x{res}_in0"] = block(cout, cout, flavor="dec", attention=True)
                self.dec[f"{res}x{res}_in1"] = block(cout, cout, flavor="dec")
            else:
                self.dec[f"{res}x{res}_up"] = block(cout, cout, flavor="dec", resample_mode="up")
            for idx in range(num_res_blocks + 1):
                cin, cout = cout + skips.pop(), channels
                self.dec[f"{res}x{res}_block{idx}"] = block(
                    cin, cout, flavor="dec", attention=res in attn_res
                )
        self.out_conv = MPConv(cout, out_channels, kernel=[3, 3])

    def convert_to_fp16(self):
        """No-op: weights stay fp32; activations use self.dtype (set at construction)."""

    def convert_to_fp32(self):
        """No-op: weights are always fp32."""

    def load_state_dict(self, state_dict, strict=True):
        if any(k.startswith(("input_blocks.", "time_embed.", "middle_block.")) for k in state_dict):
            raise RuntimeError(
                "This is an ADM (UNetModel) checkpoint, which cannot be loaded into "
                "MPUNetModel (--arch mp). The MP architecture must be trained from scratch; "
                "use --arch adm to load ADM checkpoints."
            )
        return super().load_state_dict(state_dict, strict=strict)

    def noise_labels(self, timesteps):
        """Map (possibly fractional) timesteps in [0, diffusion_steps) to log(sigma) / 4."""
        t = timesteps.float().clamp(0, len(self.c_noise_table) - 1)
        lo = t.floor().long()
        hi = (lo + 1).clamp(max=len(self.c_noise_table) - 1)
        return th.lerp(self.c_noise_table[lo], self.c_noise_table[hi], t - lo)

    def class_vectors(self, y, n, device):
        """
        Class labels -> one-hot * sqrt(num_classes). The unconditional (CFG null)
        label is the all-zero vector: pass y=None, label -1 for some entries, or a
        float [N x num_classes] tensor directly.
        """
        if y is None:
            return th.zeros(n, self.num_classes, device=device)
        if y.is_floating_point():
            vec = y.float().reshape(n, self.num_classes)
        else:
            vec = F.one_hot(y.clamp(min=0), self.num_classes).float()
            vec = vec * (y >= 0).float()[:, None]
        return vec * math.sqrt(self.num_classes)

    def forward(self, x, timesteps, y=None):
        # Embedding (fp32).
        emb = self.emb_noise(self.emb_fourier(self.noise_labels(timesteps)))
        if self.emb_label is not None:
            label = self.class_vectors(y, x.shape[0], x.device)
            emb = mp_sum(emb, self.emb_label(label), t=self.label_balance)
        emb = mp_silu(emb)

        # Encoder.
        h = th.cat([x, th.ones_like(x[:, :1])], dim=1).type(self.dtype)
        skips = []
        for name, blk in self.enc.items():
            h = blk(h) if name.endswith("_conv") else blk(h, emb)
            skips.append(h)

        # Decoder.
        for name, blk in self.dec.items():
            if "_block" in name:
                h = mp_cat(h, skips.pop(), t=self.concat_balance)
            h = blk(h, emb)
        out = self.out_conv(h, gain=self.out_gain)
        return out.type(x.dtype)
