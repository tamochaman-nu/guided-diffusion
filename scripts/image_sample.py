"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
from PIL import Image

from guided_diffusion import attributes, dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16 or args.use_bf16:
        model.convert_to_fp16()
    model.eval()

    if args.attr_cond:
        attr_labels = [
            attributes.parse_attribute("gender", args.sample_gender),
            attributes.parse_attribute("view", args.sample_view),
        ]
        logger.log(
            f"attributes: gender={args.sample_gender} view={args.sample_view} "
            f"guidance_scale={args.guidance_scale}"
        )
        model_fn = cfg_model(model, args.guidance_scale)
    else:
        model_fn = model

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        if args.attr_cond:
            model_kwargs["y"] = th.tensor(
                [attr_labels] * args.batch_size, device=dist_util.dev()
            )
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            device=dist_util.dev(),
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)
        if args.save_png:
            save_pngs(arr, os.path.join(logger.get_dir(), f"png_{shape_str}"))

    dist.barrier()
    logger.log("sampling complete")


def cfg_model(model, scale):
    """
    Classifier-free guidance on the epsilon channels: eps_u + s * (eps_c - eps_u),
    where eps_u uses the null label (-1 for every attribute). Variance channels
    (learn_sigma) are taken from the conditional output. scale=1 disables CFG.
    """
    if scale == 1:
        return model

    def fn(x, t, y=None, **kwargs):
        out_c = model(x, t, y=y, **kwargs)
        out_u = model(x, t, y=th.full_like(y, -1), **kwargs)
        c = x.shape[1]
        eps = out_u[:, :c] + scale * (out_c[:, :c] - out_u[:, :c])
        return th.cat([eps, out_c[:, c:]], dim=1)

    return fn


def save_pngs(arr, out_dir):
    """Save each sample as a PNG plus all samples tiled into grid.png."""
    os.makedirs(out_dir, exist_ok=True)
    for i, img in enumerate(arr):
        Image.fromarray(img).save(os.path.join(out_dir, f"{i:06d}.png"))
    n, h, w, c = arr.shape
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    grid = np.zeros((rows * h, cols * w, c), dtype=arr.dtype)
    for i, img in enumerate(arr):
        r, q = divmod(i, cols)
        grid[r * h : (r + 1) * h, q * w : (q + 1) * w] = img
    Image.fromarray(grid).save(os.path.join(out_dir, "grid.png"))
    logger.log(f"saved {n} PNGs and grid.png to {out_dir}")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        save_png=False,  # also write each sample (and a grid) as PNG
        # --attr_cond models: attributes to generate ("any" = unspecified) and
        # the classifier-free guidance scale (1.0 = no guidance)
        sample_gender="any",
        sample_view="any",
        guidance_scale=1.0,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
