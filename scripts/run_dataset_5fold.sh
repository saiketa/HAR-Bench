#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/benchmark:${PYTHONPATH:-}"

# Dataset-level five-fold cross-validation.
# Usage examples:
#   bash scripts/run_dataset_5fold.sh
#   MODE=TS2Vec ENCODER_BACKBONE=transformer bash scripts/run_dataset_5fold.sh
#   MODE=LIMU-BERT TRAIN_SUBSET_RATE=0.5 DATA_OTHER_SUBSET_RATE=0.3 bash scripts/run_dataset_5fold.sh

MODE="${MODE:-BioBankSSL}"
ENCODER_BACKBONE="${ENCODER_BACKBONE:-transformer}"
CLASSIFIER_PREFIX="${CLASSIFIER_PREFIX:-transformer}"
INPUT_CHANNELS="${INPUT_CHANNELS:-6}"
DATA_ROOT="${DATA_ROOT:-./data}"
DATA_OTHER_ROOT="${DATA_OTHER_ROOT:-./data_other}"
SAVE_PREFIX="${SAVE_PREFIX:-save}"
EMBED_PREFIX="${EMBED_PREFIX:-embed}"
RESULT_DIR="${RESULT_DIR:-./results_5fold}"
TRAIN_SUBSET_RATE="${TRAIN_SUBSET_RATE:-1.0}"
DATA_OTHER_SUBSET_RATE="${DATA_OTHER_SUBSET_RATE:-0.0}"
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

ALL_DATASETS=(
  DSADS HHAR HARSense
  RealWorld WISDM Motion
  PAMAP2 MHEALTH UT-Complex
  KU-HAR Shoaib UCI
  USC-HAD TNDA-HAR
)

gpu_args=()
if [[ -n "$GPU" ]]; then
  gpu_args=(-g "$GPU")
fi

fold_test_datasets() {
  case "$1" in
    1) echo "DSADS HHAR HARSense" ;;
    2) echo "RealWorld WISDM Motion" ;;
    3) echo "PAMAP2 MHEALTH UT-Complex" ;;
    4) echo "KU-HAR Shoaib UCI" ;;
    5) echo "USC-HAD TNDA-HAR" ;;
    *) echo "Unknown fold: $1" >&2; return 1 ;;
  esac
}

fold_train_datasets() {
  local fold="$1"
  local test_set=" $(fold_test_datasets "$fold") "
  local train=()
  local ds

  for ds in "${ALL_DATASETS[@]}"; do
    if [[ "$test_set" != *" $ds "* ]]; then
      train+=("$ds")
    fi
  done

  printf '%s\n' "${train[@]}"
}

mkdir -p "$RESULT_DIR"

echo "Dataset-level five-fold cross-validation"
echo "MODE=$MODE"
echo "ENCODER_BACKBONE=$ENCODER_BACKBONE"
echo "CLASSIFIER_PREFIX=$CLASSIFIER_PREFIX"
echo "DATA_ROOT=$DATA_ROOT"
echo "DATA_OTHER_ROOT=$DATA_OTHER_ROOT"
echo "TRAIN_SUBSET_RATE=$TRAIN_SUBSET_RATE"
echo "DATA_OTHER_SUBSET_RATE=$DATA_OTHER_SUBSET_RATE"
echo "CLASSIFIER_TRAIN_SUBSET_RATE=$CLASSIFIER_TRAIN_SUBSET_RATE"

for fold in 1 2 3 4 5; do
  read -r -a train_datasets <<< "$(fold_train_datasets "$fold" | tr '\n' ' ')"
  read -r -a test_datasets <<< "$(fold_test_datasets "$fold")"

  save_dir="./${SAVE_PREFIX}_${fold}"
  embed_dir="./${EMBED_PREFIX}_${fold}"
  result_csv="${RESULT_DIR}/fold_${fold}_${MODE}_${CLASSIFIER_PREFIX}_${INPUT_CHANNELS}d.csv"

  echo
  echo "================ Fold ${fold} ================"
  echo "Train datasets: ${train_datasets[*]}"
  echo "Test datasets : ${test_datasets[*]}"
  echo "Save dir      : $save_dir"
  echo "Embed dir     : $embed_dir"

  if [[ "$RUN_PRETRAIN" == "1" ]]; then
    python3 benchmark/pretrain.py \
      --mode "$MODE" \
      --encoder_backbone "$ENCODER_BACKBONE" \
      --datasets_root "$DATA_ROOT" \
      --datasets_other_root "$DATA_OTHER_ROOT" \
      -ds "${train_datasets[@]}" \
      --save_dir "$save_dir" \
      --input_channels "$INPUT_CHANNELS" \
      --train_subset_rate "$TRAIN_SUBSET_RATE" \
      --data_other_subset_rate "$DATA_OTHER_SUBSET_RATE" \
      "${gpu_args[@]}" \
      "${EXTRA_PRETRAIN_ARGS[@]}"
  fi

  if [[ "$RUN_EMBEDDING" == "1" ]]; then
    python3 benchmark/embedding.py \
      --mode "$MODE" \
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
      --mode "$MODE" \
      --classifier_prefix "$CLASSIFIER_PREFIX" \
      --save_dir "$save_dir" \
      --embed_dir "$embed_dir" \
      --input_channels "$INPUT_CHANNELS" \
      --train_subset_rate "$CLASSIFIER_TRAIN_SUBSET_RATE" \
      "${gpu_args[@]}" \
      "${EXTRA_CLASSIFIER_ARGS[@]}"
  fi

  if [[ "$RUN_TEST" == "1" ]]; then
    python3 benchmark/test.py \
      --mode "$MODE" \
      --classifier_prefix "$CLASSIFIER_PREFIX" \
      --datasets_root "$DATA_ROOT" \
      -td "${test_datasets[@]}" \
      --model_root "$save_dir" \
      --input_channels "$INPUT_CHANNELS" \
      --output_csv "$result_csv" \
      "${gpu_args[@]}" \
      "${EXTRA_TEST_ARGS[@]}"
  fi
done

echo
echo "Five-fold run finished. Fold result CSV files are under: $RESULT_DIR"
