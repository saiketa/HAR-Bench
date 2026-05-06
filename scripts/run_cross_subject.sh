#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/benchmark:${PYTHONPATH:-}"

# Cross-subject five-fold validation.
# Each dataset is split into five folds by global subject id
# (label[..., 0] = activity, label[..., 1] = global subject id).
#
# Usage examples:
#   bash scripts/run_cross_subject.sh
#   SSL_METHOD=BioBankSSL ENCODER_BACKBONE=cnn DECODER_BACKBONE=cnn bash scripts/run_cross_subject.sh
#   MODE=TS2Vec ENCODER_BACKBONE=transformer CLASSIFIER_PREFIX=cnn bash scripts/run_cross_subject.sh

SSL_METHOD="${SSL_METHOD:-${MODE:-BioBankSSL}}"
ENCODER_BACKBONE="${ENCODER_BACKBONE:-transformer}"
DECODER_BACKBONE="${DECODER_BACKBONE:-${CLASSIFIER_PREFIX:-transformer}}"
INPUT_CHANNELS="${INPUT_CHANNELS:-6}"
DATA_ROOT="${DATA_ROOT:-./data}"
SAVE_DIR="${SAVE_DIR:-./save_cross_subject}"
EMBED_DIR="${EMBED_DIR:-./embed_cross_subject}"
RESULT_DIR="${RESULT_DIR:-./results_cross_subject}"
N_FOLDS="${N_FOLDS:-5}"
PRETRAIN_TRAINING_RATE="${PRETRAIN_TRAINING_RATE:-0.8}"
CLASSIFIER_TRAINING_RATE="${CLASSIFIER_TRAINING_RATE:-0.8}"
GPU="${GPU:-}"

RUN_PRETRAIN="${RUN_PRETRAIN:-1}"
RUN_EMBEDDING="${RUN_EMBEDDING:-1}"
RUN_CLASSIFIER="${RUN_CLASSIFIER:-1}"
RUN_TEST="${RUN_TEST:-1}"

EXTRA_PRETRAIN_ARGS=(${EXTRA_PRETRAIN_ARGS:-})
EXTRA_EMBEDDING_ARGS=(${EXTRA_EMBEDDING_ARGS:-})
EXTRA_CLASSIFIER_ARGS=(${EXTRA_CLASSIFIER_ARGS:-})
EXTRA_TEST_ARGS=(${EXTRA_TEST_ARGS:-})

gpu_args=()
if [[ -n "$GPU" ]]; then
  gpu_args=(-g "$GPU")
fi

mkdir -p "$RESULT_DIR"

echo "Cross-subject five-fold validation"
echo "SSL_METHOD=$SSL_METHOD"
echo "ENCODER_BACKBONE=$ENCODER_BACKBONE"
echo "DECODER_BACKBONE=$DECODER_BACKBONE"
echo "DATA_ROOT=$DATA_ROOT"
echo "SAVE_DIR=$SAVE_DIR"
echo "EMBED_DIR=$EMBED_DIR"
echo "N_FOLDS=$N_FOLDS"

for fold_id in $(seq 0 $((N_FOLDS - 1))); do
  result_csv="${RESULT_DIR}/fold_${fold_id}_${SSL_METHOD}_${ENCODER_BACKBONE}_${DECODER_BACKBONE}_${INPUT_CHANNELS}d.csv"

  echo
  echo "================ Subject Fold ${fold_id}/${N_FOLDS} ================"

  if [[ "$RUN_PRETRAIN" == "1" ]]; then
    python3 benchmark/pretrain_subject_cv.py \
      --mode "$SSL_METHOD" \
      --encoder_backbone "$ENCODER_BACKBONE" \
      --fold_id "$fold_id" \
      --n_folds "$N_FOLDS" \
      --datasets_root "$DATA_ROOT" \
      --save_dir "$SAVE_DIR" \
      --embed_dir "$EMBED_DIR" \
      --input_channels "$INPUT_CHANNELS" \
      --training_rate "$PRETRAIN_TRAINING_RATE" \
      "${gpu_args[@]}" \
      "${EXTRA_PRETRAIN_ARGS[@]}"
  fi

  if [[ "$RUN_EMBEDDING" == "1" ]]; then
    python3 benchmark/embedding_subject_cv.py \
      --mode "$SSL_METHOD" \
      --encoder_backbone "$ENCODER_BACKBONE" \
      --fold_id "$fold_id" \
      --n_folds "$N_FOLDS" \
      --datasets_root "$DATA_ROOT" \
      --save_dir "$SAVE_DIR" \
      --embed_dir "$EMBED_DIR" \
      --input_channels "$INPUT_CHANNELS" \
      "${gpu_args[@]}" \
      "${EXTRA_EMBEDDING_ARGS[@]}"
  fi

  if [[ "$RUN_CLASSIFIER" == "1" ]]; then
    python3 benchmark/classifier_subject_cv.py \
      --mode "$SSL_METHOD" \
      --method "$DECODER_BACKBONE" \
      --fold_id "$fold_id" \
      --n_folds "$N_FOLDS" \
      --datasets_root "$DATA_ROOT" \
      --save_dir "$SAVE_DIR" \
      --embed_dir "$EMBED_DIR" \
      --input_channels "$INPUT_CHANNELS" \
      --training_rate "$CLASSIFIER_TRAINING_RATE" \
      "${gpu_args[@]}" \
      "${EXTRA_CLASSIFIER_ARGS[@]}"
  fi

  if [[ "$RUN_TEST" == "1" ]]; then
    python3 benchmark/test_subject_cv.py \
      --mode "$SSL_METHOD" \
      --encoder_backbone "$ENCODER_BACKBONE" \
      --classifier_prefix "$DECODER_BACKBONE" \
      --fold_id "$fold_id" \
      --n_folds "$N_FOLDS" \
      --datasets_root "$DATA_ROOT" \
      --save_dir "$SAVE_DIR" \
      --input_channels "$INPUT_CHANNELS" \
      --output_csv "$result_csv" \
      "${gpu_args[@]}" \
      "${EXTRA_TEST_ARGS[@]}"
  fi
done

echo
echo "Cross-subject five-fold run finished. Results are under: $RESULT_DIR"
