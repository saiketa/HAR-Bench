#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/benchmark:${PYTHONPATH:-}"

# Cross-device validation between custom-device and research-device datasets.
#
# Terminology:
#   custom   = daily-device datasets from the legacy scripts
#   research = experiment-device datasets from the legacy scripts
#
# Usage examples:
#   bash scripts/run_cross_device.sh
#   SSL_METHOD=BioBankSSL ENCODER_BACKBONE=cnn DECODER_BACKBONE=cnn MODALITY=acc_gyro bash scripts/run_cross_device.sh
#   MODE=TS2Vec ENCODER_BACKBONE=transformer CLASSIFIER_PREFIX=cnn MODALITY=acc DIRECTION=custom_to_research bash scripts/run_cross_device.sh

SSL_METHOD="${SSL_METHOD:-${MODE:-BioBankSSL}}"
ENCODER_BACKBONE="${ENCODER_BACKBONE:-transformer}"
DECODER_BACKBONE="${DECODER_BACKBONE:-${CLASSIFIER_PREFIX:-transformer}}"
MODALITY="${MODALITY:-acc_gyro}"
INPUT_CHANNELS="${INPUT_CHANNELS:-}"
DATA_ROOT="${DATA_ROOT:-./data}"
SAVE_ROOT="${SAVE_ROOT:-./save_cross_device}"
EMBED_ROOT="${EMBED_ROOT:-./embed_cross_device}"
RESULT_DIR="${RESULT_DIR:-./results_cross_device}"
TRAIN_SUBSET_RATE="${TRAIN_SUBSET_RATE:-1.0}"
CLASSIFIER_TRAIN_SUBSET_RATE="${CLASSIFIER_TRAIN_SUBSET_RATE:-1.0}"
DIRECTION="${DIRECTION:-both}"
GPU="${GPU:-}"

RUN_PRETRAIN="${RUN_PRETRAIN:-1}"
RUN_EMBEDDING="${RUN_EMBEDDING:-1}"
RUN_CLASSIFIER="${RUN_CLASSIFIER:-1}"
RUN_TEST="${RUN_TEST:-1}"

EXTRA_PRETRAIN_ARGS=(${EXTRA_PRETRAIN_ARGS:-})
EXTRA_EMBEDDING_ARGS=(${EXTRA_EMBEDDING_ARGS:-})
EXTRA_CLASSIFIER_ARGS=(${EXTRA_CLASSIFIER_ARGS:-})
EXTRA_TEST_ARGS=(${EXTRA_TEST_ARGS:-})

CUSTOM_DATASETS=(
  HARSense
  HHAR
  KU-HAR
  Motion
  RealWorld
  Shoaib
  UCI
  UT-Complex
  WISDM
)

RESEARCH_DATASETS=(
  DSADS
  MHEALTH
  PAMAP2
  TNDA-HAR
  USC-HAD
)

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

run_one_direction() {
  local train_domain="$1"
  local test_domain="$2"
  shift 2

  local train_datasets=()
  local test_datasets=()

  if [[ "$train_domain" == "custom" ]]; then
    train_datasets=("${CUSTOM_DATASETS[@]}")
  else
    train_datasets=("${RESEARCH_DATASETS[@]}")
  fi

  if [[ "$test_domain" == "custom" ]]; then
    test_datasets=("${CUSTOM_DATASETS[@]}")
  else
    test_datasets=("${RESEARCH_DATASETS[@]}")
  fi

  local run_name="${train_domain}_to_${test_domain}"
  local save_dir="${SAVE_ROOT}/${run_name}"
  local embed_dir="${EMBED_ROOT}/${run_name}"
  local result_csv="${RESULT_DIR}/${run_name}_${SSL_METHOD}_${ENCODER_BACKBONE}_${DECODER_BACKBONE}_${INPUT_CHANNELS}d.csv"

  echo
  echo "================ Cross-device: ${train_domain} -> ${test_domain} ================"
  echo "Train datasets: ${train_datasets[*]}"
  echo "Test datasets : ${test_datasets[*]}"
  echo "Save dir      : $save_dir"
  echo "Embed dir     : $embed_dir"
  echo "Result CSV    : $result_csv"

  if [[ "$RUN_PRETRAIN" == "1" ]]; then
    python3 benchmark/pretrain.py \
      --mode "$SSL_METHOD" \
      --encoder_backbone "$ENCODER_BACKBONE" \
      --datasets_root "$DATA_ROOT" \
      -ds "${train_datasets[@]}" \
      --save_dir "$save_dir" \
      --input_channels "$INPUT_CHANNELS" \
      --train_subset_rate "$TRAIN_SUBSET_RATE" \
      --data_other_subset_rate 0.0 \
      "${gpu_args[@]}" \
      "${EXTRA_PRETRAIN_ARGS[@]}"
  fi

  if [[ "$RUN_EMBEDDING" == "1" ]]; then
    python3 benchmark/embedding.py \
      --mode "$SSL_METHOD" \
      --encoder_backbone "$ENCODER_BACKBONE" \
      --datasets_root "$DATA_ROOT" \
      -ds "${train_datasets[@]}" \
      --save_dir "$save_dir" \
      --embed_dir "$embed_dir" \
      --input_channels "$INPUT_CHANNELS" \
      "${gpu_args[@]}" \
      "${EXTRA_EMBEDDING_ARGS[@]}"
  fi

  if [[ "$RUN_CLASSIFIER" == "1" ]]; then
    python3 benchmark/classifier.py \
      --mode "$SSL_METHOD" \
      --classifier_prefix "$DECODER_BACKBONE" \
      --save_dir "$save_dir" \
      --embed_dir "$embed_dir" \
      --input_channels "$INPUT_CHANNELS" \
      --train_subset_rate "$CLASSIFIER_TRAIN_SUBSET_RATE" \
      "${gpu_args[@]}" \
      "${EXTRA_CLASSIFIER_ARGS[@]}"
  fi

  if [[ "$RUN_TEST" == "1" ]]; then
    python3 benchmark/test.py \
      --mode "$SSL_METHOD" \
      --encoder_backbone "$ENCODER_BACKBONE" \
      --classifier_prefix "$DECODER_BACKBONE" \
      --datasets_root "$DATA_ROOT" \
      -td "${test_datasets[@]}" \
      --model_root "$save_dir" \
      --input_channels "$INPUT_CHANNELS" \
      --output_csv "$result_csv" \
      "${gpu_args[@]}" \
      "${EXTRA_TEST_ARGS[@]}"
  fi
}

echo "Cross-device validation"
echo "SSL_METHOD=$SSL_METHOD"
echo "ENCODER_BACKBONE=$ENCODER_BACKBONE"
echo "DECODER_BACKBONE=$DECODER_BACKBONE"
echo "MODALITY=$MODALITY"
echo "INPUT_CHANNELS=$INPUT_CHANNELS"
echo "DATA_ROOT=$DATA_ROOT"
echo "DIRECTION=$DIRECTION"
echo "TRAIN_SUBSET_RATE=$TRAIN_SUBSET_RATE"
echo "CLASSIFIER_TRAIN_SUBSET_RATE=$CLASSIFIER_TRAIN_SUBSET_RATE"

case "$DIRECTION" in
  both)
    run_one_direction custom research
    run_one_direction research custom
    ;;
  custom_to_research)
    run_one_direction custom research
    ;;
  research_to_custom)
    run_one_direction research custom
    ;;
  *)
    echo "Unsupported DIRECTION=$DIRECTION. Use both, custom_to_research, or research_to_custom." >&2
    exit 1
    ;;
esac

echo
echo "Cross-device run finished. Results are under: $RESULT_DIR"
