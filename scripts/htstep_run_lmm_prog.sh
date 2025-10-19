#!/bin/bash
#SBATCH --job-name=htstep_run_lmm_prog
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/output_htstep_run_lmm_prog_%j.log
#SBATCH --error=logs/output_htstep_run_lmm_prog_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128GB
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --account=<REPLACE_WITH_ACCOUNT_NAME>
#SBATCH --partition=<REPLACE_WITH_PARTITION_NAME>

export HF_HOME="<REPLACE_WITH_PATH_TO_HF_CACHE>"
export HF_DATASETS_OFFLINE=1
export HF_TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=32
export LD_LIBRARY_PATH="$HOME/opt/ffmpeg-n7.1-latest-linux64-gpl-shared-7.1/lib:$LD_LIBRARY_PATH"

VENV_DIR="<REPLACE_WITH_BAGLM_VENV_PATH>"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

DATASET_DIR="<REPLACE_WITH_HTSTEP_DIR>"
ANNOTATIONS_JSON="$DATASET_DIR/data/annotations.json"
TAXONOMY_CSV="$DATASET_DIR/data/taxonomy.csv"
VIDEO_ANNOTS_FILE="datasets/htstep/htstep_video_annots.json"

MODEL="internvl2.5-8b"
SPLIT="val_seen"
PROMPT_TYPE="prog"
QUESTION_FILE="prompts/$PROMPT_TYPE/question.txt"

RESULT_DIR="$DATASET_DIR/results/$PROMPT_TYPE/"
mkdir -p "$RESULT_DIR"
cp "$QUESTION_FILE" "$RESULT_DIR"

python src/htstep_eval.py \
    --dataset_dir "$DATASET_DIR" \
    --model "$MODEL" \
    --annotations_json "$ANNOTATIONS_JSON" \
    --taxonomy_csv "$TAXONOMY_CSV" \
    --split "$SPLIT" \
    --visual_batch_size 16 \
    --text_batch_size 1 \
    --segment_duration 2 \
    --sampling_fps 2 \
    --question_file "$QUESTION_FILE" \
    --num_workers 0 \
    --video_annots_file "$VIDEO_ANNOTS_FILE" \
    --prompt_type "$PROMPT_TYPE" \
    --result_dir "$RESULT_DIR"
