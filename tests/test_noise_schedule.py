"""
Tests for the shifted cosine noise schedule ("cosine_shift<k>").

Run:  python tests/test_noise_schedule.py        (or: pytest tests/test_noise_schedule.py)
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guided_diffusion.gaussian_diffusion import get_named_beta_schedule  # noqa: E402


def log_snr(name, steps=1000):
    a = np.cumprod(1.0 - get_named_beta_schedule(name, steps))
    return np.log(a / (1 - a))


def test_shift1_equals_cosine():
    assert np.array_equal(
        get_named_beta_schedule("cosine_shift1", 1000), get_named_beta_schedule("cosine", 1000)
    )


def test_log_snr_is_shifted():
    for k in (2, 4):
        diff = log_snr(f"cosine_shift{k}") - log_snr("cosine")
        # Away from the clipped last steps (max_beta=0.999) the shift is constant; it
        # differs from -2 log k by ~5e-4 (k=2) to ~2e-3 (k=4) because betas_for_alpha_bar normalizes by
        # alpha_bar(0) != 1 (inherited from the original cosine discretization).
        mid = diff[:950]
        print(f"    k={k}: logSNR shift {mid.min():.6f} .. {mid.max():.6f} (expected {-2 * math.log(k):.6f})")
        assert np.ptp(mid) < 1e-6 and np.allclose(mid, -2 * math.log(k), atol=5e-3)


def test_high_noise_share():
    for name in ("cosine", "cosine_shift4"):
        eq = log_snr(name) + 2 * math.log(4)  # logSNR of the image downsampled 256 -> 64 px
        print(f"    {name:14s} share of timesteps with 64px-equivalent logSNR < 0: {np.mean(eq < 0):.1%}")
    assert np.mean(log_snr("cosine_shift4") + 2 * math.log(4) < 0) > 0.45


def test_mp_noise_labels_follow_schedule():
    from guided_diffusion.unet_mp import MPUNetModel

    def table(schedule):
        m = MPUNetModel(64, 3, 32, 6, 1, (), channel_mult=(1, 2), noise_schedule=schedule)
        return m.c_noise_table.numpy()

    d = table("cosine_shift4") - table("cosine")
    print(f"    MP c_noise shift {d[:950].mean():.4f} (expected log(4)/4 = {math.log(4) / 4:.4f}), "
          f"range {table('cosine_shift4')[0]:.3f} .. {table('cosine_shift4')[-1]:.3f}")
    assert np.allclose(d[:950], math.log(4) / 4, atol=1e-3)


if __name__ == "__main__":
    failed = 0
    for name, fn in [(k, v) for k, v in globals().items() if k.startswith("test_")]:
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {name}: {e!r}")
    sys.exit(1 if failed else 0)
