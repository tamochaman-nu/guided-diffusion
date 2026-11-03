"""
Tests for attribute (gender / view) conditioning with metadata.jsonl.

Run:  python tests/test_attr_cond.py        (or: pytest tests/test_attr_cond.py)
"""

import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch as th
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from guided_diffusion import attributes, logger  # noqa: E402
from guided_diffusion.image_datasets import load_data  # noqa: E402
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults  # noqa: E402
from guided_diffusion.train_util import TrainLoop  # noqa: E402

DEV = "cuda" if th.cuda.is_available() else "cpu"
CFG = dict(image_size=32, num_channels=32, num_res_blocks=1, num_head_channels=32,
           attention_resolutions="8", channel_mult="1,2", learn_sigma=True,
           diffusion_steps=1000, noise_schedule="cosine_shift4", arch="mp", attr_cond=True)


def build(**kw):
    a = model_and_diffusion_defaults()
    a.update(CFG, **kw)
    model, diffusion = create_model_and_diffusion(**a)
    return model.to(DEV), diffusion


def make_dataset(root, n=12):
    """Images in root (and a subdir) plus a manifest covering most of them."""
    os.makedirs(os.path.join(root, "sub"), exist_ok=True)
    recs, rng = [], np.random.RandomState(0)
    for i in range(n):
        rel = f"sub/{i:03d}.png" if i % 2 else f"{i:03d}.png"
        Image.fromarray(rng.randint(0, 255, (40, 40, 3), dtype=np.uint8)).save(os.path.join(root, rel))
        if i == n - 1:
            continue  # no entry -> unknown
        g = ["female", "male", "unknown"][i % 3]
        v = ["front", "three_quarter", "profile", "unknown"][i % 4]
        recs.append({"file": rel, "gender": g, "view": v})
    with open(os.path.join(root, "metadata.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    return recs


def test_encode_and_tag_fallback():
    assert attributes.encode({"gender": "male", "view": "profile"}) == [1, 2]
    assert attributes.encode({"gender": "unknown", "view": "front"}) == [-1, 0]
    # explicit fields win over tags
    assert attributes.encode({"gender": "female", "tags": {"1boy": 0.9}}) == [0, -1]
    # fallback from tagger output (dict with confidences or list)
    assert attributes.encode({"tags": {"1boy": 0.9, "profile": 0.7, "1girl": 0.2}}) == [1, 2]
    assert attributes.encode({"tags": ["1girl", "looking_at_viewer"]}) == [0, -1]
    # several characters / both genders -> unknown
    assert attributes.encode({"tags": ["1girl", "1boy"]}) == [-1, -1]
    assert attributes.encode({"tags": ["1girl", "multiple_girls"]}) == [-1, -1]
    try:
        attributes.encode({"file": "x", "gender": "woman"})
    except ValueError:
        pass
    else:
        raise AssertionError("invalid value accepted")


def test_manifest_and_loader():
    with tempfile.TemporaryDirectory() as root:
        recs = make_dataset(root)
        files = sorted(os.path.join(root, r["file"]) for r in recs)
        files.append(os.path.join(root, "sub", "011.png"))
        labels = attributes.load_manifest(os.path.join(root, "metadata.jsonl"), files, root)
        expect = {os.path.join(root, r["file"]): attributes.encode(r) for r in recs}
        for p, lab in zip(files, labels):
            assert lab == expect.get(p, [-1, -1]), (p, lab)
        print("    summary:", attributes.summarize(labels))
        data = load_data(data_dir=root, batch_size=4, image_size=32, deterministic=True,
                         attr_file=os.path.join(root, "metadata.jsonl"))
        x, cond = next(data)
        assert x.shape == (4, 3, 32, 32) and cond["y"].shape == (4, 2) and cond["y"].dtype == th.int64
        print("    batch y:", cond["y"].tolist())


def test_label_vectors_unit_rms_and_null():
    model, _ = build()
    assert model.emb_label.weight.shape[1] == sum(attributes.ATTR_DIMS) == 5
    y = th.tensor([[0, 2], [1, 0], [-1, 1], [-1, -1]], device=DEV)
    v = model.class_vectors(y, 4, DEV)
    rms = v.square().mean(1).sqrt().tolist()
    print(f"    label vector RMS (full, full, partial, null): {[round(r, 3) for r in rms]}")
    assert abs(rms[0] - 1) < 1e-6 and abs(rms[1] - 1) < 1e-6
    assert abs(rms[2] - math.sqrt(0.5)) < 1e-6 and rms[3] == 0
    assert th.equal(model.class_vectors(None, 4, DEV)[0], v[3])


def test_cfg_training_and_label_effect():
    model, diffusion = build()

    def data():
        g = th.Generator().manual_seed(0)
        while True:
            yield th.rand(8, 3, 32, 32, generator=g) * 2 - 1, {
                "y": th.randint(-1, 3, (8, 2), generator=g).clamp(max=1)}

    loop = TrainLoop(model=model, diffusion=diffusion, data=data(), batch_size=8, microbatch=8,
                     lr=0.01, ema_rate="0.999", log_interval=100, save_interval=10**9,
                     resume_checkpoint="", optimizer="adam", adam_betas=(0.9, 0.99),
                     lr_schedule="edm2", cond_drop_prob=0.3, attr_drop_prob=0.3)
    th.manual_seed(0)
    y = th.zeros(10000, 2, dtype=th.long, device=DEV)
    d = loop._drop_labels(y)
    all_null = (d == -1).all(1).float().mean().item()
    one = (d == -1).float().mean().item()
    print(f"    dropout: both null {all_null:.3f} (expected {0.3 + 0.7 * 0.09:.3f}), "
          f"per attribute {one:.3f} (expected {0.3 + 0.7 * 0.3:.3f})")
    assert abs(all_null - 0.363) < 0.02 and abs(one - 0.51) < 0.02
    assert (y == 0).all()  # input not modified
    for _ in range(5):
        batch, cond = next(loop.data)
        loop.run_step(batch, cond)
    model.eval()
    x = th.randn(2, 3, 32, 32, device=DEV)
    t = th.tensor([500, 500], device=DEV)
    with th.no_grad():
        a = model(x, t, y=th.tensor([[0, 0], [0, 0]], device=DEV))
        b = model(x, t, y=th.tensor([[1, 2], [1, 2]], device=DEV))
        n = model(x, t, y=th.tensor([[-1, -1], [-1, -1]], device=DEV))
    assert th.isfinite(a).all() and not th.equal(a, b) and not th.equal(a, n)
    print("    outputs differ by label after training:", (a - b).abs().max().item())


def test_cfg_sampling_wrapper():
    from image_sample import cfg_model

    model, _ = build()
    th.nn.init.constant_(model.out_gain, 1.0)
    for blk in model.modules():
        if hasattr(blk, "emb_gain"):
            th.nn.init.constant_(blk.emb_gain, 0.5)
    model.eval()
    x = th.randn(2, 3, 32, 32, device=DEV)
    t = th.tensor([100, 800], device=DEV)
    y = th.tensor([[1, 2], [0, -1]], device=DEV)
    with th.no_grad():
        assert cfg_model(model, 1.0) is model
        out = cfg_model(model, 3.0)(x, t, y=y)
        c, u = model(x, t, y=y), model(x, t, y=th.full_like(y, -1))
    assert th.allclose(out[:, :3], u[:, :3] + 3.0 * (c[:, :3] - u[:, :3]), atol=1e-5)
    assert th.equal(out[:, 3:], c[:, 3:])
    print("    CFG eps = u + s(c - u); variance channels from the conditional output")


def test_adm_rejects_attr_cond():
    try:
        build(arch="adm")
    except AssertionError as e:
        print("    adm:", e)
    else:
        raise AssertionError("attr_cond with arch=adm was accepted")


def test_cli_train_and_sample():
    with tempfile.TemporaryDirectory() as root:
        make_dataset(os.path.join(root, "data"), n=16)
        env = dict(os.environ, OPENAI_LOGDIR=os.path.join(root, "log"), PYTHONPATH=REPO)
        env.pop("ACT_MONITOR", None)
        flags = ["--image_size", "32", "--num_channels", "32", "--num_res_blocks", "1",
                 "--num_head_channels", "32", "--attention_resolutions", "8", "--channel_mult", "1,2",
                 "--learn_sigma", "True", "--noise_schedule", "cosine_shift4", "--arch", "mp",
                 "--attr_cond", "True"]
        r = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "image_train.py"),
                            "--data_dir", os.path.join(root, "data"), "--batch_size", "4",
                            "--lr_anneal_steps", "6", "--save_interval", "1000", "--cond_drop_prob", "0.1",
                            "--attr_drop_prob", "0.1", "--mp_lr_warmup_steps", "2", *flags],
                           env=env, capture_output=True, text=True, cwd=root)
        log = open(os.path.join(root, "log", "log.txt")).read()
        assert r.returncode == 0, r.stderr[-2000:]
        assert "attribute labels from" in log, log
        print("    train log:", [l for l in log.splitlines() if "attribute labels" in l][0][:160])
        env["OPENAI_LOGDIR"] = os.path.join(root, "sample")
        r = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "image_sample.py"),
                            "--model_path", os.path.join(root, "log", "ema_0.9999_000006.pt"),
                            "--num_samples", "4", "--batch_size", "4", "--timestep_respacing", "10",
                            "--sample_gender", "male", "--sample_view", "profile",
                            "--guidance_scale", "2.0", "--save_png", "True", *flags],
                           env=env, capture_output=True, text=True, cwd=root)
        assert r.returncode == 0, r.stderr[-2000:]
        assert os.path.exists(os.path.join(root, "sample", "png_4x32x32x3", "grid.png"))
        print("    sampled 4 images with gender=male view=profile guidance=2.0")


if __name__ == "__main__":
    logger.configure(dir=tempfile.mkdtemp(), format_strs=[])
    from guided_diffusion import dist_util

    dist_util.setup_dist()
    failed = 0
    for name, fn in [(k, v) for k, v in list(globals().items()) if k.startswith("test_")]:
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            failed += 1
            print(f"[FAIL] {name}: {e!r}")
    sys.exit(1 if failed else 0)
