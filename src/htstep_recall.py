import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="internvl2.5-8b", type=str)
    parser.add_argument("--question", default=None, type=str)
    parser.add_argument("--lmm_vsg_dir", required=True, type=str)
    parser.add_argument(
        "--video_annots_file", default="datasets/htstep/htstep_video_annots.json", type=str
    )
    return parser.parse_args()


def get_Y_pred(scores):
    binary_matrix = np.zeros_like(scores, dtype=int)
    binary_matrix[np.argmax(scores, axis=0), np.arange(scores.shape[1])] = 1
    return binary_matrix


def get_Y_true(T, K, video_annot):
    Y_true = np.zeros([T, K], dtype=np.uint8)
    # steps = act2var2steps[video_annot['activity']][video_annot['variation']]
    steps = video_annot["step_headline"]
    step_to_k = {step: k for k, step in enumerate(steps)}

    for clip in video_annot["clips"]:
        for annotation in clip["annotations"]:
            for lang_query in annotation["language_queries"]:
                step = lang_query["query"]
                k = step_to_k[step]
                start_t, end_t = lang_query["clip_start_sec"], lang_query["clip_end_sec"]
                Y_true[math.floor(start_t) : math.ceil(end_t) + 1, k] = 1
    return Y_true


# def get_Y_true(T, K, video_annot):
#     Y_true = np.zeros([T, K], dtype=np.uint8)
#     video_end_sec = T

#     # steps = set()
#     queries = []
#     annotations = []

#     for clip in video_annot["clips"]:
#         for annotation in clip["annotations"]:
#             for lang_query in annotation["language_queries"]:
#                 # steps.add(lang_query["step"])
#                 queries.append(lang_query["query"])
#                 annotations.append((lang_query["query"], lang_query["clip_start_sec"], lang_query["clip_end_sec"]))

#     queries = list(dict.fromkeys(queries))

#     # Y_true.setdefault(task, {})
#     # Y_true[task].setdefault(video_uid, np.zeros((video_end_sec, len(queries)), dtype=int))

#     for query, start, end in annotations:
#         start_t, end_t = int(start), min(int(end), video_end_sec)
#         if start_t < video_end_sec:
#             query_idx = queries.index(query)
#             Y_true[start_t:end_t, query_idx] = 1
#             # Y_true[task][video_uid][start_t:end_t, query_idx] = 1

#     return Y_true


def get_recall_by_total_candidate_steps(Y_true, Y_pred):
    """Computes step recall grouped by the total number of candidate steps (K).

    Returns:
        num_steps (K) -> recall info
    """
    recall_by_K = defaultdict(list)

    for task in Y_true:
        ys_true = Y_true[task]
        ys_pred = Y_pred[task]

        for vid in set(ys_pred.keys()).intersection(set(ys_true.keys())):
            y_true = ys_true[vid]
            y_pred = ys_pred[vid]
            T, K = y_true.shape
            for k in range(K):
                if y_true[:, k].sum() > 0:
                    pred_t = np.argmax(y_pred[:, k])
                    if y_true[pred_t, k] == 1:
                        recall_by_K[K].append(1)
                    else:
                        recall_by_K[K].append(0)

    return {
        K: {"mean_recall": np.mean(recalls), "count": len(recalls)}
        for K, recalls in recall_by_K.items()
    }


def get_recall(Y_true, Y_pred):
    # step_match = 0
    # step_total = 0
    recall = []
    for task in Y_true:
        ys_true = Y_true[task]
        ys_pred = Y_pred[task]

        for vid in set(ys_pred.keys()).intersection(set(ys_true.keys())):
            y_true = ys_true[vid]
            y_pred = ys_pred[vid]
            K = y_true.shape[1]
            for k in range(K):
                if y_true[:, k].sum() > 0:
                    # step_total += 1
                    pred_t = np.argmax(y_pred[:, k])
                    if y_true[pred_t, k] == 1:
                        recall.append(1)
                    else:
                        recall.append(0)
                        # step_match += 1
                    # step_match += (y_true[:, k] * y_pred[:, k]).sum()
                    # step_total += 1
    # return step_match / step_total if step_total > 0 else 0
    return np.mean(recall)


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

    Y_true = {}
    Y_pred = {}

    for video_annot in video_annots:
        video_uid = video_annot["video_uid"]
        task = video_annot["activity"]
        result_path = os.path.join(lmm_vsg_dir, f"{video_uid}.pt")
        scores = torch.load(result_path)

        if task not in Y_pred:
            Y_pred[task] = {}

        y_pred = get_Y_pred(scores.numpy())
        Y_pred[task][video_uid] = y_pred

        if task not in Y_true:
            Y_true[task] = {}
        Y_true[task][video_uid] = get_Y_true(*y_pred.shape, video_annot)

    recall = get_recall(Y_true, Y_pred)
    print(f"Recall: {recall:.4f}")


if __name__ == "__main__":
    main()
