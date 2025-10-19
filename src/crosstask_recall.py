import argparse
import json
import math
import os
import random

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="internvl2.5-8b", type=str)
    parser.add_argument("--question", default=None, type=str)
    parser.add_argument("--lmm_vsg_dir", required=True, type=str)
    parser.add_argument(
        "--video_annots_file", default="datasets/crosstask/crosstask_video_annots.json", type=str
    )
    return parser.parse_args()


def get_Y_pred(scores):
    binary_matrix = np.zeros_like(scores, dtype=int)
    binary_matrix[np.argmax(scores, axis=0), np.arange(scores.shape[1])] = 1
    return binary_matrix


def get_Y_true(T, K, video_annot):
    Y_true = np.zeros([T, K], dtype=np.uint8)
    steps = video_annot["task_steps"]
    step_to_k = {step: k for k, step in enumerate(steps)}

    for clip in video_annot["clips"]:
        for annotation in clip["annotations"]:
            for lang_query in annotation["language_queries"]:
                step = lang_query["query"]
                k = step_to_k[step]
                start_t, end_t = lang_query["clip_start_sec"], lang_query["clip_end_sec"]
                Y_true[math.floor(start_t) : math.ceil(end_t) + 1, k] = 1
    return Y_true


def get_recall(Y_true, Y_pred):
    step_match = {task: 0 for task in Y_true.keys()}
    step_total = {task: 0 for task in Y_true.keys()}
    for task, ys_true in Y_true.items():
        ys_pred = Y_pred[task]
        for vid in set(ys_pred.keys()).intersection(set(ys_true.keys())):
            y_true = ys_true[vid]
            y_pred = ys_pred[vid]
            step_total[task] += (y_true.sum(axis=0) > 0).sum()
            step_match[task] += (y_true * y_pred).sum()
    recalls = {task: step_match[task] / n for task, n in step_total.items()}
    return recalls


def main():
    args = parse_args()
    lmm_vsg_dir = os.path.join(args.lmm_vsg_dir, args.model)
    print(f"Performance of {args.model}.")

    with open(args.video_annots_file) as f:
        video_annots = json.load(f)

    len_video_annots = len(video_annots)
    video_annots = [
        vid
        for vid in video_annots
        if os.path.exists(os.path.join(lmm_vsg_dir, f"{vid['video_uid']}.pt"))
    ]
    assert len_video_annots == len(video_annots)
    print(len(video_annots))

    num_sets = 20
    set_size = 1850

    recalls_all_sets = []

    for set_idx in range(num_sets):
        random.seed(set_idx)
        # print(f"Processing set {set_idx+1}/{num_sets}...")
        sampled_videos = random.sample(video_annots, min(set_size, len(video_annots)))

        Y_true = {}
        Y_pred = {}

        for video_annot in sampled_videos:
            video_uid = video_annot["video_uid"]
            task = video_annot["task_name"]
            scores_path = os.path.join(lmm_vsg_dir, f"{video_uid}.pt")
            scores = torch.load(scores_path)

            if task not in Y_pred:
                Y_pred[task] = {}
            y_pred = get_Y_pred(scores.numpy())
            Y_pred[task][video_uid] = y_pred

            if task not in Y_true:
                Y_true[task] = {}
            Y_true[task][video_uid] = get_Y_true(*y_pred.shape, video_annot)

        recalls = get_recall(Y_true, Y_pred)
        avg_recall = np.mean(list(recalls.values()))
        recalls_all_sets.append(avg_recall)
        # print(f"  Set {set_idx+1} average recall: {avg_recall:.4f}")

    final_avg_recall = np.mean(recalls_all_sets)

    print(f"Recall: {final_avg_recall:.4f}")


if __name__ == "__main__":
    main()
