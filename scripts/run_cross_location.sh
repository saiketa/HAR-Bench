#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/benchmark:${PYTHONPATH:-}"

# Cross-location five-fold validation.
# Labels are expected to use:
#   label[..., 0] = activity id
#   label[..., 1] = global subject id
#   label[..., 2] = global body-location id
#
# Location groups follow the original HAR-Bench body-part grouping:
#   trunk = 0, 8, 9, 12
#   upper = 1, 6, 11
#   lower = 2, 3, 4, 5, 7
#
# Usage examples:
#   TRAIN_LOCATION=trunk TEST_LOCATION=upper bash scripts/run_cross_location.sh
#   SSL_METHOD=BioBankSSL ENCODER_BACKBONE=cnn DECODER_BACKBONE=cnn TRAIN_LOCATION=trunk TEST_LOCATION=lower bash scripts/run_cross_location.sh

SSL_METHOD="${SSL_METHOD:-${MODE:-BioBankSSL}}"
ENCODER_BACKBONE="${ENCODER_BACKBONE:-transformer}"
DECODER_BACKBONE="${DECODER_BACKBONE:-${CLASSIFIER_PREFIX:-transformer}}"
TRAIN_LOCATION="${TRAIN_LOCATION:-trunk}"
TEST_LOCATION="${TEST_LOCATION:-upper}"
MODALITY="${MODALITY:-acc_gyro}"
INPUT_CHANNELS="${INPUT_CHANNELS:-}"
DATA_ROOT="${DATA_ROOT:-./data}"
SAVE_DIR="${SAVE_DIR:-./save_cross_location}"
EMBED_DIR="${EMBED_DIR:-./embed_cross_location}"
RESULT_DIR="${RESULT_DIR:-./results_cross_location}"
N_FOLDS="${N_FOLDS:-5}"
PRETRAIN_TRAINING_RATE="${PRETRAIN_TRAINING_RATE:-0.8}"
CLASSIFIER_TRAINING_RATE="${CLASSIFIER_TRAINING_RATE:-0.8}"
CLASSIFIER_TRAIN_SUBSET_RATE="${CLASSIFIER_TRAIN_SUBSET_RATE:-1.0}"
GPU="${GPU:-}"

RUN_PRETRAIN="${RUN_PRETRAIN:-1}"
RUN_EMBEDDING="${RUN_EMBEDDING:-1}"
RUN_CLASSIFIER="${RUN_CLASSIFIER:-1}"
RUN_TEST="${RUN_TEST:-1}"

EXTRA_PRETRAIN_ARGS=(${EXTRA_PRETRAIN_ARGS:-})
EXTRA_EMBEDDING_ARGS=(${EXTRA_EMBEDDING_ARGS:-})
EXTRA_CLASSIFIER_ARGS=(${EXTRA_CLASSIFIER_ARGS:-})
EXTRA_TEST_ARGS=(${EXTRA_TEST_ARGS:-})

if [[ -z "$INPUT_CHANNELS" ]]; then
  case "$MODALITY" in
    acc|accel|accelerometer)
      INPUT_CHANNELS=3
      ;;
    acc_gyro|acc+gyro|imu|all)
      INPUT_CHANNELS=6
      ;;
    *)
      echo "Unsupported MODALITY=$MODALITY. Use acc or acc_gyro, or set INPUT_CHANNELS directly." >&2
      exit 1
      ;;
  esac
fi

gpu_args=()
if [[ -n "$GPU" ]]; then
  gpu_args=(-g "$GPU")
fi

mkdir -p "$RESULT_DIR"
result_csv="${RESULT_DIR}/train_${TRAIN_LOCATION}_test_${TEST_LOCATION}_${SSL_METHOD}_${ENCODER_BACKBONE}_${DECODER_BACKBONE}_${INPUT_CHANNELS}d.csv"
rm -f "$result_csv"

echo "Cross-location five-fold validation"
echo "SSL_METHOD=$SSL_METHOD"
echo "ENCODER_BACKBONE=$ENCODER_BACKBONE"
echo "DECODER_BACKBONE=$DECODER_BACKBONE"
echo "TRAIN_LOCATION=$TRAIN_LOCATION"
echo "TEST_LOCATION=$TEST_LOCATION"
echo "MODALITY=$MODALITY"
echo "INPUT_CHANNELS=$INPUT_CHANNELS"
echo "DATA_ROOT=$DATA_ROOT"
echo "N_FOLDS=$N_FOLDS"

for fold_id in $(seq 0 $((N_FOLDS - 1))); do
  echo
  echo "================ Location Fold ${fold_id}/${N_FOLDS}: ${TRAIN_LOCATION} -> ${TEST_LOCATION} ================"

  if [[ "$RUN_PRETRAIN" == "1" ]]; then
    python3 benchmark/pretrain_cross_location.py \
      --mode "$SSL_METHOD" \
      --encoder_backbone "$ENCODER_BACKBONE" \
      --fold_id "$fold_id" \
      --n_folds "$N_FOLDS" \
      --train_location "$TRAIN_LOCATION" \
      --test_location "$TEST_LOCATION" \
      --datasets_root "$DATA_ROOT" \
      --save_dir "$SAVE_DIR" \
      --embed_dir "$EMBED_DIR" \
      --input_channels "$INPUT_CHANNELS" \
      --training_rate "$PRETRAIN_TRAINING_RATE" \
      "${gpu_args[@]}" \
      "${EXTRA_PRETRAIN_ARGS[@]}"
  fi

  if [[ "$RUN_EMBEDDING" == "1" ]]; then
    python3 benchmark/embedding_cross_location.py \
      --mode "$SSL_METHOD" \
      --encoder_backbone "$ENCODER_BACKBONE" \
      --fold_id "$fold_id" \
      --n_folds "$N_FOLDS" \
      --train_location "$TRAIN_LOCATION" \
      --test_location "$TEST_LOCATION" \
      --datasets_root "$DATA_ROOT" \
      --save_dir "$SAVE_DIR" \
      --embed_dir "$EMBED_DIR" \
      --input_channels "$INPUT_CHANNELS" \
      "${gpu_args[@]}" \
      "${EXTRA_EMBEDDING_ARGS[@]}"
  fi

  if [[ "$RUN_CLASSIFIER" == "1" ]]; then
    python3 benchmark/classifier_cross_location.py \
      --mode "$SSL_METHOD" \
      --method "$DECODER_BACKBONE" \
      --fold_id "$fold_id" \
      --n_folds "$N_FOLDS" \
      --train_location "$TRAIN_LOCATION" \
      --test_location "$TEST_LOCATION" \
      --datasets_root "$DATA_ROOT" \
      --save_dir "$SAVE_DIR" \
      --embed_dir "$EMBED_DIR" \
      --input_channels "$INPUT_CHANNELS" \
      --training_rate "$CLASSIFIER_TRAINING_RATE" \
      --label_rate "$CLASSIFIER_TRAIN_SUBSET_RATE" \
      "${gpu_args[@]}" \
      "${EXTRA_CLASSIFIER_ARGS[@]}"
  fi

  if [[ "$RUN_TEST" == "1" ]]; then
    python3 benchmark/test_cross_location.py \
      --mode "$SSL_METHOD" \
      --encoder_backbone "$ENCODER_BACKBONE" \
      --classifier_prefix "$DECODER_BACKBONE" \
      --fold_id "$fold_id" \
      --n_folds "$N_FOLDS" \
      --train_location "$TRAIN_LOCATION" \
      --test_location "$TEST_LOCATION" \
      --datasets_root "$DATA_ROOT" \
      --save_dir "$SAVE_DIR" \
      --embed_dir "$EMBED_DIR" \
      --input_channels "$INPUT_CHANNELS" \
      --output_csv "$result_csv" \
      "${gpu_args[@]}" \
      "${EXTRA_TEST_ARGS[@]}"
  fi
done

echo
echo "Cross-location run finished. Results saved to: $result_csv"
