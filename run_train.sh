#!/bin/bash
set -e
cd /app

export OPENAI_LOGDIR="${TRAIN_LOGDIR:-/app/logs/train}"
mkdir -p "$OPENAI_LOGDIR"

MODEL_FLAGS="--image_size ${IMAGE_SIZE:-64} --num_channels ${NUM_CHANNELS:-128} \
--num_res_blocks ${NUM_RES_BLOCKS:-3} --num_heads ${NUM_HEADS:-4} \
--num_head_channels ${NUM_HEAD_CHANNELS:--1} \
--attention_resolutions ${ATTENTION_RESOLUTIONS:-16,8} \
--dropout ${DROPOUT:-0.1} --learn_sigma ${LEARN_SIGMA:-True} \
--class_cond ${CLASS_COND:-False} --use_scale_shift_norm ${USE_SCALE_SHIFT_NORM:-True} \
--resblock_updown ${RESBLOCK_UPDOWN:-False} \
--use_new_attention_order ${USE_NEW_ATTENTION_ORDER:-False} \
--use_checkpoint ${USE_CHECKPOINT:-False} --use_fp16 ${USE_FP16:-False} --use_bf16 ${USE_BF16:-False} \
--arch ${ARCH:-adm} --mp_clip_act ${MP_CLIP_ACT:-256}"
[ -n "$CHANNEL_MULT" ] && MODEL_FLAGS="$MODEL_FLAGS --channel_mult $CHANNEL_MULT"

DIFFUSION_FLAGS="--diffusion_steps ${DIFFUSION_STEPS:-1000} --noise_schedule ${NOISE_SCHEDULE:-cosine}"

TRAIN_FLAGS="--lr ${LR:-1e-4} --weight_decay ${WEIGHT_DECAY:-0.0} \
--batch_size ${BATCH_SIZE:-16} --microbatch ${MICROBATCH:--1} \
--lr_anneal_steps ${LR_ANNEAL_STEPS:-0} --ema_rate ${EMA_RATE:-0.9999} \
--log_interval ${LOG_INTERVAL:-10} --save_interval ${SAVE_INTERVAL:-10000} \
--mp_lr_ref ${MP_LR_REF:-0.01} --mp_lr_ref_steps ${MP_LR_REF_STEPS:-70000} \
--mp_lr_warmup_steps ${MP_LR_WARMUP_STEPS:-5000}"

# 再開: RESUME_CHECKPOINT 指定を優先、なければ最新の modelNNNNNN.pt を自動検出
RESUME="$RESUME_CHECKPOINT"
if [ -z "$RESUME" ]; then
  RESUME=$(ls "$OPENAI_LOGDIR"/model[0-9]*.pt 2>/dev/null | sort -V | tail -n 1 || true)
fi
[ -n "$RESUME" ] && TRAIN_FLAGS="$TRAIN_FLAGS --resume_checkpoint $RESUME"

echo "resume: ${RESUME:-none}"
python scripts/image_train.py --data_dir /app/data \
  $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS $EXTRA_TRAIN_FLAGS
