#!/bin/bash
set -e
cd /app

# モデル・拡散過程・学習のパラメータ (cfgs/*.cfg。環境変数の値が優先)
. scripts/load_cfg.sh
load_cfg "${TRAIN_CFG:?TRAIN_CFG (cfgs/*.cfg) is not set}"

export OPENAI_LOGDIR="${TRAIN_LOGDIR:-/app/logs/train}"
mkdir -p "$OPENAI_LOGDIR"
# 同じログディレクトリで別のモデル・スケジュールを学習 (再開) しないよう、前回の設定と照合してから保存する
check_cfg_compatible "$OPENAI_LOGDIR/train.cfg"
save_cfg "$OPENAI_LOGDIR/train.cfg"

MODEL_FLAGS="--image_size ${IMAGE_SIZE:-64} --num_channels ${NUM_CHANNELS:-128} \
--num_res_blocks ${NUM_RES_BLOCKS:-3} --num_heads ${NUM_HEADS:-4} \
--num_head_channels ${NUM_HEAD_CHANNELS:--1} \
--attention_resolutions ${ATTENTION_RESOLUTIONS:-16,8} \
--dropout ${DROPOUT:-0.1} --learn_sigma ${LEARN_SIGMA:-True} \
--class_cond ${CLASS_COND:-False} --use_scale_shift_norm ${USE_SCALE_SHIFT_NORM:-True} \
--resblock_updown ${RESBLOCK_UPDOWN:-False} \
--use_new_attention_order ${USE_NEW_ATTENTION_ORDER:-False} \
--use_checkpoint ${USE_CHECKPOINT:-False} --use_fp16 ${USE_FP16:-False} --use_bf16 ${USE_BF16:-False} \
--arch ${ARCH:-adm} --mp_clip_act ${MP_CLIP_ACT:-256} --attr_cond ${ATTR_COND:-False}"
[ -n "$CHANNEL_MULT" ] && MODEL_FLAGS="$MODEL_FLAGS --channel_mult $CHANNEL_MULT"

DIFFUSION_FLAGS="--diffusion_steps ${DIFFUSION_STEPS:-1000} --noise_schedule ${NOISE_SCHEDULE:-cosine}"

TRAIN_FLAGS="--lr ${LR:-1e-4} --weight_decay ${WEIGHT_DECAY:-0.0} \
--batch_size ${BATCH_SIZE:-16} --microbatch ${MICROBATCH:--1} \
--lr_anneal_steps ${LR_ANNEAL_STEPS:-0} --ema_rate ${EMA_RATE:-0.9999} \
--log_interval ${LOG_INTERVAL:-10} --save_interval ${SAVE_INTERVAL:-10000} \
--mp_lr_ref ${MP_LR_REF:-0.01} --mp_lr_ref_steps ${MP_LR_REF_STEPS:-70000} \
--mp_lr_warmup_steps ${MP_LR_WARMUP_STEPS:-5000} \
--cond_drop_prob ${COND_DROP_PROB:-0.0} --attr_drop_prob ${ATTR_DROP_PROB:-0.0}"
[ -n "$ATTR_FILE" ] && TRAIN_FLAGS="$TRAIN_FLAGS --attr_file $ATTR_FILE"

# 再開: RESUME_CHECKPOINT 指定を優先、なければ最新の modelNNNNNN.pt を自動検出
RESUME="$RESUME_CHECKPOINT"
if [ -z "$RESUME" ]; then
  RESUME=$(ls "$OPENAI_LOGDIR"/model[0-9]*.pt 2>/dev/null | sort -V | tail -n 1 || true)
fi
[ -n "$RESUME" ] && TRAIN_FLAGS="$TRAIN_FLAGS --resume_checkpoint $RESUME"

# 1 つの学習を複数 GPU で並列化 (MPI + DDP)。既定はコンテナに見えている GPU の枚数
# (.env の GPU_ANIME=0,1 などで決まる)。BATCH_SIZE は GPU 1 枚あたりの値で、
# 全体のバッチは BATCH_SIZE x NUM_GPUS になる
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)}
[ "${NUM_GPUS:-0}" -ge 1 ] || NUM_GPUS=1
if [ "$NUM_GPUS" -gt 1 ]; then
  LAUNCH="mpiexec -n $NUM_GPUS --bind-to none --mca btl_vader_single_copy_mechanism none python"
else
  LAUNCH="python"
fi

echo "resume: ${RESUME:-none} / gpus: $NUM_GPUS"
$LAUNCH scripts/image_train.py --data_dir /app/data \
  $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS $EXTRA_TRAIN_FLAGS
