#!/bin/bash
set -e
cd /app

export OPENAI_LOGDIR="${SAMPLE_LOGDIR:-/app/logs/sample}"
mkdir -p "$OPENAI_LOGDIR"

# MODEL_PATH 未指定なら TRAIN_LOGDIR の最新 ema を使う
MODEL="$MODEL_PATH"
if [ -z "$MODEL" ]; then
  MODEL=$(ls "${TRAIN_LOGDIR:-/app/logs/train}"/ema_*.pt 2>/dev/null | sort -V | tail -n 1 || true)
fi
[ -z "$MODEL" ] && { echo "MODEL_PATH not set and no ema checkpoint found" >&2; exit 1; }

MODEL_FLAGS="--image_size ${IMAGE_SIZE:-64} --num_channels ${NUM_CHANNELS:-128} \
--num_res_blocks ${NUM_RES_BLOCKS:-3} --num_heads ${NUM_HEADS:-4} \
--num_head_channels ${NUM_HEAD_CHANNELS:--1} \
--attention_resolutions ${ATTENTION_RESOLUTIONS:-16,8} \
--dropout ${DROPOUT:-0.1} --learn_sigma ${LEARN_SIGMA:-True} \
--class_cond ${CLASS_COND:-False} --use_scale_shift_norm ${USE_SCALE_SHIFT_NORM:-True} \
--resblock_updown ${RESBLOCK_UPDOWN:-False} \
--use_new_attention_order ${USE_NEW_ATTENTION_ORDER:-False} \
--use_fp16 ${USE_FP16:-False} --use_bf16 ${USE_BF16:-False}"
[ -n "$CHANNEL_MULT" ] && MODEL_FLAGS="$MODEL_FLAGS --channel_mult $CHANNEL_MULT"

python scripts/image_sample.py --model_path "$MODEL" $MODEL_FLAGS \
  --diffusion_steps ${DIFFUSION_STEPS:-1000} --noise_schedule ${NOISE_SCHEDULE:-cosine} \
  --timestep_respacing "${TIMESTEP_RESPACING}" \
  --num_samples ${NUM_SAMPLES:-64} --batch_size ${SAMPLE_BATCH_SIZE:-16} \
  --use_ddim ${USE_DDIM:-False}
