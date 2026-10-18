"""
Tests for the EDM2 magnitude-preserving U-Net (guided_diffusion/unet_mp.py).

Run:  python tests/test_unet_mp.py        (or: pytest tests/test_unet_mp.py)
The ADM regression test compares against the git revision MP_TEST_ADM_REF
(default: HEAD, i.e. the code before uncommitted changes); set it to the commit
before the MP change once that is committed.
"""

import io
import json
import math
import os
import subprocess
import sys
import tempfile

import torch as th

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from guided_diffusion import logger  # noqa: E402
from guided_diffusion.fp16_util import MixedPrecisionTrainer  # noqa: E402
from guided_diffusion.script_util import (  # noqa: E402
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from guided_diffusion.unet import UNetModel  # noqa: E402
from guided_diffusion.unet_mp import (  # noqa: E402
    MPBlock,
    MPConv,
    MPFourier,
    MPUNetModel,
    mp_cat,
    mp_silu,
    mp_sum,
    normalize,
)

DEV = "cuda" if th.cuda.is_available() else "cpu"

# The training configuration currently used in docker-compose.yml.
TRAIN_CFG = dict(
    image_size=256, num_channels=64, num_res_blocks=2, num_heads=4, num_head_channels=64,
    attention_resolutions="32,16,8", dropout=0.0, learn_sigma=True, use_scale_shift_norm=True,
    resblock_updown=True, use_checkpoint=True, diffusion_steps=1000, noise_schedule="cosine",
)
SMALL_CFG = dict(TRAIN_CFG, image_size=64, num_channels=32, num_head_channels=32,
                 attention_resolutions="16,8")


def build(cfg=SMALL_CFG, seed=0, **kw):
    th.manual_seed(seed)
    args = model_and_diffusion_defaults()
    args.update(cfg, arch="mp", **kw)
    model, diffusion = create_model_and_diffusion(**args)
    return model.to(DEV), diffusion


def rms(x):
    return x.float().square().mean().sqrt().item()


def assert_close_to_one(name, value, tol=0.1):
    print(f"    {name:40s} rms={value:.4f}")
    assert abs(value - 1) <= tol, f"{name}: rms {value:.4f} not within 1 +- {tol}"


def mpconvs(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, MPConv)]


def train_steps(model, diffusion, use_fp16=False, lr=0.01, cond=None, steps=3):
    """
    Optimizer steps through MixedPrecisionTrainer (as TrainLoop does). Note: with
    out_gain = 0 at init, only out_gain receives a gradient in the first step, so
    the other weights start moving from the second step.
    """
    model.train()
    mp = MixedPrecisionTrainer(model=model, use_fp16=use_fp16)
    opt = th.optim.Adam(mp.master_params, lr=lr, betas=(0.9, 0.99))
    taken = tries = 0
    while taken < steps:  # fp16 skips steps while the dynamic loss scale settles
        x = th.randn(2, 3, model.image_size, model.image_size, device=DEV)
        t = th.randint(0, 1000, (2,), device=DEV)
        mp.zero_grad()
        loss = diffusion.training_losses(model, x, t, model_kwargs=cond or {})["loss"].mean()
        mp.backward(loss)
        taken += mp.optimize(opt)
        tries += 1
        assert tries < 30, "too many skipped (non-finite) steps"
    mp.n_skipped = tries - taken
    return mp


# 1. Each MP operation keeps unit RMS for N(0,1) inputs.
def test_mp_ops_unit_rms():
    th.manual_seed(0)
    x = th.randn(8, 64, 32, 32, device=DEV)
    assert_close_to_one("MPConv 3x3 (64->128)", rms(MPConv(64, 128, [3, 3]).to(DEV)(x)))
    assert_close_to_one("MPConv 1x1 (64->32)", rms(MPConv(64, 32, [1, 1]).to(DEV)(x)))
    assert_close_to_one(
        "MPConv fully-connected (256->512)",
        rms(MPConv(256, 512, []).to(DEV)(th.randn(4096, 256, device=DEV))),
    )
    assert_close_to_one("mp_silu", rms(mp_silu(x)))
    for t in (0.3, 0.5):
        assert_close_to_one(f"mp_sum(t={t})", rms(mp_sum(x, th.randn_like(x), t=t)))
    a, b = th.randn(8, 32, 16, 16, device=DEV), th.randn(8, 96, 16, 16, device=DEV)
    c = mp_cat(a, b, t=0.5)
    assert_close_to_one("mp_cat (32 + 96 channels)", rms(c))
    ea, eb = c[:, :32].float().square().sum().item(), c[:, 32:].float().square().sum().item()
    print(f"    mp_cat contribution a/b energy ratio = {ea / eb:.4f} (expected 1)")
    assert abs(ea / eb - 1) < 0.1
    fourier = MPFourier(256).to(DEV)
    assert_close_to_one("MPFourier (input N(0,1))", rms(fourier(th.randn(4096, device=DEV))))
    model, _ = build()
    c_noise = model.noise_labels(th.arange(1000, device=DEV))
    assert_close_to_one("MPFourier (all 1000 timesteps)", rms(fourier(c_noise)))
    assert_close_to_one("PixelNorm normalize(5x, dim=1)", rms(normalize(5 * x, dim=1)))
    xh = (5 * x).half()
    assert_close_to_one("PixelNorm fp16 input", rms(normalize(xh, dim=1)))


# 2. At initialization every block output has O(1) RMS.
def test_init_block_rms():
    model, _ = build(TRAIN_CFG)
    model.eval()
    stats = []
    hooks = [
        blk.register_forward_hook(
            lambda m, i, o, name=f"{side}.{key}": stats.append((name, tuple(o.shape[1:]), rms(o)))
        )
        for side in ("enc", "dec")
        for key, blk in getattr(model, side).items()
    ]
    x = th.randn(2, 3, 256, 256, device=DEV)
    t = th.tensor([10, 900], device=DEV)
    with th.no_grad():
        out = model(x, t)
    for h in hooks:
        h.remove()
    print(f"    {'block':24s} {'shape':18s} rms")
    for name, shape, r in stats:
        print(f"    {name:24s} {str(shape):18s} {r:.4f}")
    assert all(0.2 < r < 5 for _, _, r in stats), "block RMS not O(1)"
    assert rms(out) == 0


# 3. After a training step, raw MPConv weights stay at norm sqrt(fan_in) per output
#    channel (forced weight normalization), also with the fp16 trainer path.
def test_forced_weight_norm_after_step():
    for use_fp16 in (False, True):
        model, diffusion = build(use_fp16=use_fp16)
        w0 = {n: m.weight.detach().clone() for n, m in mpconvs(model)}
        mp = train_steps(model, diffusion, use_fp16=use_fp16)
        assert not mp.flat_master and mp.master_params[0] is next(model.parameters())
        moved = max((m.weight.detach() - w0[n]).abs().max().item() for n, m in mpconvs(model))
        assert moved > 1e-3, "weights did not move"

        def max_dev():
            dev = 0.0
            for _, m in mpconvs(model):
                w = m.weight.detach().float()
                assert w.dtype == th.float32 and m.weight.dtype == th.float32
                norms = w.flatten(1).norm(dim=1)
                dev = max(dev, (norms / math.sqrt(w[0].numel()) - 1).abs().max().item())
            return dev

        after_step = max_dev()
        model.train()
        with th.no_grad():  # the next training forward re-normalizes
            model(th.randn(1, 3, 64, 64, device=DEV), th.tensor([5], device=DEV))
        after_fwd = max_dev()
        print(
            f"    use_fp16={use_fp16}: max |norm/sqrt(fan_in) - 1| right after the step "
            f"= {after_step:.2e}, after next forward = {after_fwd:.2e} "
            f"({len(mpconvs(model))} MPConv, max weight change {moved:.3f}, "
            f"skipped fp16 steps {mp.n_skipped})"
        )
        # normalize() divides by (eps + rms) with eps = 1e-4, so norms are
        # sqrt(fan_in) / (1 + 1e-4) by construction (same as EDM2).
        assert after_fwd < 2e-4


# 4. out_gain = emb_gain = 0 at init, so the initial output is exactly 0.
def test_initial_output_is_zero():
    model, _ = build()
    assert model.out_gain.item() == 0
    assert all(b.emb_gain.item() == 0 for b in model.modules() if isinstance(b, MPBlock))
    out = model(th.randn(3, 3, 64, 64, device=DEV), th.tensor([0, 500, 999], device=DEV))
    assert out.shape == (3, 6, 64, 64)
    assert th.count_nonzero(out) == 0
    print("    output identically 0:", out.abs().max().item())


# 5. state_dict save/load reproduces the output exactly (incl. Fourier buffers).
def test_state_dict_roundtrip():
    model, diffusion = build(seed=0)
    train_steps(model, diffusion)  # make the gains non-zero
    buf = io.BytesIO()
    th.save(model.state_dict(), buf)
    buf.seek(0)
    model2, _ = build(seed=123)
    model2.load_state_dict(th.load(buf, map_location=DEV, weights_only=True))
    model.eval()
    model2.eval()
    x = th.randn(2, 3, 64, 64, device=DEV)
    t = th.tensor([3, 700], device=DEV)
    with th.no_grad():
        a, b = model(x, t), model2(x, t)
    print(f"    max |out| = {a.abs().max().item():.4g}, identical = {th.equal(a, b)}")
    assert a.abs().max() > 0 and th.equal(a, b)


ADM_SCRIPT = r"""
import json, random, sys
import numpy as np, torch as th
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults
from guided_diffusion.train_util import TrainLoop
dist_util.setup_dist()
th.backends.cudnn.benchmark = False
th.backends.cudnn.deterministic = True
dev = "cuda" if th.cuda.is_available() else "cpu"
logger.configure(dir=sys.argv[2], format_strs=["csv"])
use_fp16 = sys.argv[1] == "fp16"
random.seed(0); np.random.seed(0); th.manual_seed(0)
a = model_and_diffusion_defaults()
a.update(image_size=64, num_channels=32, num_res_blocks=2, num_heads=4,
         attention_resolutions="16,8", dropout=0.1, learn_sigma=True, use_scale_shift_norm=True,
         resblock_updown=True, use_checkpoint=True, use_fp16=use_fp16,
         diffusion_steps=1000, noise_schedule="cosine")
model, diffusion = create_model_and_diffusion(**a)
model.to(dev)
g = th.Generator().manual_seed(1)
def data():
    while True:
        yield th.rand(4, 3, 64, 64, generator=g) * 2 - 1, {}
loop = TrainLoop(model=model, diffusion=diffusion, data=data(), batch_size=4, microbatch=2,
                 lr=1e-4, ema_rate="0.9999", log_interval=1, save_interval=10**9,
                 resume_checkpoint="", use_fp16=use_fp16, weight_decay=0.01, lr_anneal_steps=100)
losses = []
for i in range(4):
    batch, cond = next(loop.data)
    loop.run_step(batch, cond)
    losses.append(float(logger.getkvs()["loss"]))
    logger.dumpkvs()
    loop.step += 1
model.eval()
with th.no_grad():
    out = model(th.linspace(-1, 1, 3 * 64 * 64 * 2, device=dev).reshape(2, 3, 64, 64),
                th.tensor([10, 600], device=dev))
sd = loop.mp_trainer.master_params_to_state_dict(loop.mp_trainer.master_params)
print("RESULT " + json.dumps(dict(
    type=type(model).__name__, losses=losses,
    out=[float(out.double().sum()), float(out.double().abs().sum())],
    params=float(sum(v.double().abs().sum() for v in sd.values())),
    lr=loop.opt.param_groups[0]["lr"], opt=type(loop.opt).__name__)))
"""


def _run_adm(pythonpath, mode):
    with tempfile.TemporaryDirectory() as logdir:
        env = dict(os.environ, PYTHONPATH=pythonpath, CUBLAS_WORKSPACE_CONFIG=":4096:8")
        env.pop("ACT_MONITOR", None)
        r = subprocess.run(
            [sys.executable, "-c", ADM_SCRIPT, mode, logdir],
            env=env, capture_output=True, text=True, cwd=logdir,
        )
    line = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
    assert line, r.stdout + r.stderr
    return json.loads(line[0][7:])


# 6. --arch adm is unchanged: outputs and training match the reference revision.
def test_adm_regression():
    ref = os.environ.get("MP_TEST_ADM_REF", "HEAD")
    with tempfile.TemporaryDirectory() as tmp:
        archive = subprocess.run(
            ["git", "-C", REPO, "archive", ref, "guided_diffusion"], capture_output=True
        )
        assert archive.returncode == 0, archive.stderr.decode()
        subprocess.run(["tar", "-x", "-C", tmp], input=archive.stdout, check=True)
        for mode in ("fp32", "fp16"):
            before = _run_adm(tmp, mode)
            after = _run_adm(REPO, mode)
            print(f"    {mode}: ref={ref} {before}")
            print(f"    {mode}: now     {after}")
            assert before == after, f"ADM ({mode}) differs from {ref}"


# 7. CFG null label (all-zero label vector) works, and arch defaults stay ADM.
def test_null_label_and_misc():
    model, diffusion = build(class_cond=True)
    model.eval()
    x = th.randn(4, 3, 64, 64, device=DEV)
    t = th.tensor([1, 2, 3, 4], device=DEV)
    y = th.tensor([3, -1, 7, -1], device=DEV)
    train_steps(model, diffusion, cond=dict(y=y[:2]))  # trains with null labels too
    model.eval()
    with th.no_grad():
        out_y = model(x, t, y=y)
        out_none = model(x, t, y=None)
        out_vec = model(x, t, y=th.zeros(4, model.num_classes, device=DEV))
    assert th.isfinite(out_y).all()
    assert th.equal(out_y[1], out_none[1]) and th.equal(out_y[3], out_none[3])
    assert th.equal(out_none, out_vec)
    assert not th.equal(out_y[0], out_none[0])
    print("    null label (-1) == y=None == zero vector; labelled rows differ")

    # ADM checkpoints are rejected with a clear error.
    args = model_and_diffusion_defaults()
    args.update(SMALL_CFG)
    adm, _ = create_model_and_diffusion(**args)
    assert isinstance(adm, UNetModel)
    mp_model, _ = build()
    try:
        mp_model.load_state_dict(adm.state_dict())
    except RuntimeError as e:
        print("    ADM checkpoint ->", str(e)[:80], "...")
    else:
        raise AssertionError("loading an ADM checkpoint did not fail")
    assert isinstance(mp_model, MPUNetModel)


def main():
    logger.configure(dir=tempfile.mkdtemp(), format_strs=[])
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = []
    for name, fn in tests:
        print(f"[RUN ] {name}")
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print(f"[FAIL] {name}: {e}")
            failed.append(name)
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed" + (f"; failed: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
