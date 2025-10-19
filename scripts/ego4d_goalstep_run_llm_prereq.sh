#!/bin/bash
#SBATCH --job-name=ego4d_goalstep_run_llm_prereq
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/output_ego4d_goalstep_run_llm_prereq_%j.log
#SBATCH --error=logs/output_ego4d_goalstep_run_llm_prereq_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=400GB
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --account=<REPLACE_WITH_ACCOUNT_NAME>
#SBATCH --partition=<REPLACE_WITH_PARTITION_NAME>

export HF_HOME="<REPLACE_WITH_PATH_TO_HF_CACHE>"
export HF_DATASETS_OFFLINE=1
export HF_TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=32

VENV_DIR="<REPLACE_WITH_BAGLM_VENV_PATH>"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

DATASET_DIR="<REPLACE_WITH_EGO4D_DATA_DIR>"
ANNOT_DIR="$DATASET_DIR/v2/annotations"

PROMPT_TYPE="prereq"
QUESTION_FILE="prompts/$PROMPT_TYPE/question.txt"
SYSTEM_FILE="prompts/$PROMPT_TYPE/system.txt"
MODEL="llama-3.3-70b-instruct"

RESULT_DIR="$DATASET_DIR/results/$PROMPT_TYPE/"
mkdir -p "$RESULT_DIR"
cp "$QUESTION_FILE" "$RESULT_DIR"

python src/ego4d_goalstep_qa.py \
    --dataset_dir "$DATASET_DIR" \
    --annot_dir "$ANNOT_DIR" \
    --model "$MODEL" \
    --batch_size 64 \
    --question_file "$QUESTION_FILE" \
    --system_file "$SYSTEM_FILE" \
    --num_workers 0 \
    --result_dir "$RESULT_DIR"
