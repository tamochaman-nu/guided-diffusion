"""
Helpers for various likelihood-based losses. These are ported from the original
Ho et al. diffusion models codebase:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py
"""

import numpy as np

import torch as th
import torch.nn.functional as F


def wavelet_texture_loss(pred_x0, target_x0, level=2):
    """
    ウェーブレット高周波成分のMSEロスでテクスチャ学習を強化する。
    pytorch_wavelets が使える場合はHaar DWT、なければLaplacianフィルタでフォールバック。
    """
    try:
        from pytorch_wavelets import DWTForward
        xfm = DWTForward(J=level, mode="zero", wave="haar").to(pred_x0.device)
        # fp16対応: float32にキャストしてからDWT適用
        pred_f = pred_x0.float()
        tgt_f = target_x0.float()
        _, pred_high = xfm(pred_f)
        _, tgt_high = xfm(tgt_f)
        loss = sum(F.mse_loss(p, t) for p, t in zip(pred_high, tgt_high))
        return loss / level
    except ImportError:
        # Laplacianフィルタで高周波成分を近似。
        # カーネルの二乗ノルム ||L||_F^2 = 20 で割ることで DWT と同スケール (O(1)) にする。
        _LAP_NORM_SQ = 20.0

        def high_freq(x):
            lap = th.tensor(
                [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
                dtype=x.dtype, device=x.device,
            )
            lap = lap.view(1, 1, 3, 3).expand(x.shape[1], -1, -1, -1)
            return F.conv2d(x, lap, padding=1, groups=x.shape[1])

        return F.mse_loss(high_freq(pred_x0), high_freq(target_x0)) / _LAP_NORM_SQ


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = th.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs
