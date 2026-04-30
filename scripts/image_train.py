"""
Train a diffusion model on images.
"""

import argparse

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

    device = dist_util.dev()
    if device.type == "cuda":
        import torch as th
        gpu_name = th.cuda.get_device_name(device)
        gpu_mem = th.cuda.get_device_properties(device).total_memory // (1024 ** 2)
        logger.log(f"using GPU: {gpu_name} ({gpu_mem} MiB)")
    else:
        logger.log("using CPU (CUDA not available)")

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        use_cfg=args.use_cfg,
        domain_id=args.domain_id,
        p_uncond=args.p_uncond,
        null_domain_id=args.null_domain_id,
        texture_bias_alpha=args.texture_bias_alpha,
        texture_bias_max_t_frac=args.texture_bias_max_t_frac,
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
        # CFG
        domain_id=0,          # 0=FFHQ, 1=anime
        p_uncond=0.15,        # unconditional dropout rate
        null_domain_id=2,     # null domain index (= num_domains)
        # texture bias
        texture_bias_alpha=1.0,   # 1.0=uniform, >1.0 biases toward small t
        texture_bias_max_t_frac=0.5,  # t の上限を T*frac に制限 (0.5=T/2)
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
