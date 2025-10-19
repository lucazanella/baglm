#!/bin/bash

export HF_HOME="<REPLACE_WITH_PATH_TO_HF_CACHE>"
export HF_DATASETS_OFFLINE=1
export HF_TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=32

VENV_DIR="<REPLACE_WITH_BAGLM_VENV_PATH>"
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

DATASET_DIR="<REPLACE_WITH_EGO4D_DATA_DIR>"
VIDEO_ANNOTS_FILE="datasets/ego4d_goalstep/ego4d_goalstep_video_annots.json"

MODEL="internvl2.5-8b"

LMM_VSG_DIR="$DATASET_DIR/results/vsg/"
LMM_PROG_DIR="$DATASET_DIR/results/prog/"
LLM_PREREQ_DIR="$DATASET_DIR/results/prereq/"

python src/bayes_filter.py \
    --dataset "ego4d_goalstep" \
    --model "$MODEL" \
    --video_annots_file "$VIDEO_ANNOTS_FILE" \
    --lmm_vsg_dir "$LMM_VSG_DIR" \
    --lmm_prog_dir "$LMM_PROG_DIR" \
    --llm_prereq_dir "$LLM_PREREQ_DIR"
