#!/bin/bash
#SBATCH --job-name=ego4d_goalstep_run_lmm_vsg
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/output_ego4d_goalstep_run_lmm_vsg_%j.log
#SBATCH --error=logs/output_ego4d_goalstep_run_lmm_vsg_%j.log
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

DATASET_DIR="<REPLACE_WITH_EGO4D_DATA_DIR>"
ANNOT_DIR="$DATASET_DIR/v2/annotations"
VIDEO_DIR="$DATASET_DIR/v2/full_scale"
VIDEO_ANNOTS_FILE="datasets/ego4d_goalstep/ego4d_goalstep_video_annots.json"

MODEL="internvl2.5-8b"
PROMPT_TYPE="vsg"
QUESTION_FILE="prompts/$PROMPT_TYPE/question.txt"

RESULT_DIR="$DATASET_DIR/results/$PROMPT_TYPE/"
mkdir -p "$RESULT_DIR"
cp "$QUESTION_FILE" "$RESULT_DIR"

python src/ego4d_goalstep_eval.py \
    --dataset_dir "$DATASET_DIR" \
    --annot_dir "$ANNOT_DIR" \
    --video_dir "$VIDEO_DIR" \
    --model "$MODEL" \
    --visual_batch_size 16 \
    --text_batch_size 1 \
    --segment_duration 2 \
    --sampling_fps 2 \
    --question_file "$QUESTION_FILE" \
    --num_workers 0 \
    --video_annots_file "$VIDEO_ANNOTS_FILE" \
    --prompt_type "$PROMPT_TYPE" \
    --result_dir "$RESULT_DIR"
