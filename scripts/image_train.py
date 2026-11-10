"""
Train a diffusion model on images.
"""

import argparse
import os

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    attr_file = None
    if args.attr_cond:
        attr_file = args.attr_file or os.path.join(args.data_dir, "metadata.jsonl")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        attr_file=attr_file,
    )

    if args.arch == "mp":
        # EDM2 training setup: Adam(0.9, 0.99) without weight decay and the
        # inverse-sqrt lr schedule; --lr and --weight_decay are not used.
        logger.log(
            f"arch=mp: Adam lr_ref={args.mp_lr_ref} t_ref={args.mp_lr_ref_steps} "
            f"warmup={args.mp_lr_warmup_steps} (ignoring --lr/--weight_decay)"
        )
        opt_kwargs = dict(
            lr=args.mp_lr_ref,
            weight_decay=0.0,
            optimizer="adam",
            adam_betas=(0.9, 0.99),
            lr_schedule="edm2",
            lr_ref_steps=args.mp_lr_ref_steps,
            lr_warmup_steps=args.mp_lr_warmup_steps,
        )
    else:
        opt_kwargs = dict(lr=args.lr, weight_decay=args.weight_decay)

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        use_bf16=args.use_bf16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        lr_anneal_steps=args.lr_anneal_steps,
        cond_drop_prob=args.cond_drop_prob,
        attr_drop_prob=args.attr_drop_prob,
        **opt_kwargs,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        # --arch mp (EDM2) lr schedule: mp_lr_ref / sqrt(max(step / mp_lr_ref_steps, 1))
        # with a linear warmup. EDM2's defaults (lr_ref 0.01, t_ref 70k iterations,
        # warmup 10M images ~ 5k iterations) assume batch size 2048; scale them to
        # the actual batch size and training length (see cfgs/*.cfg).
        mp_lr_ref=0.01,
        mp_lr_ref_steps=70000,
        mp_lr_warmup_steps=5000,
        # --attr_cond: labels from attr_file (default <data_dir>/metadata.jsonl).
        # CFG training drops the whole label with cond_drop_prob and each attribute
        # with attr_drop_prob (so e.g. only gender can be specified at sampling).
        attr_file="",
        cond_drop_prob=0.0,
        attr_drop_prob=0.0,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
