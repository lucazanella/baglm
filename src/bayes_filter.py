import argparse
import importlib
import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from utils.crosstask_utils import filter_nested_dict_by_video_uids


def build_successor_matrix(dependency_matrix: torch.Tensor) -> torch.Tensor:
    S = dependency_matrix.shape[0]
    mat = dependency_matrix.T.clone()
    zero_sum_cols = mat.sum(dim=0) == 0
    mat[:, zero_sum_cols] = 1.0
    mat.fill_diagonal_(1.0)
    mat = torch.cat([mat, torch.ones(S, 1)], dim=1)  # add none column
    mat = torch.cat([mat, torch.ones(1, S + 1)], dim=0)  # add none row
    return mat / mat.sum(dim=1, keepdim=True)


def safe_normalize(x: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    s = x.sum(dim=dim, keepdim=(dim is not None))
    s[s == 0] = 1.0
    return x / s


def compute_readiness_validity(
    dep_mat: torch.Tensor, progress_vec: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    row_sum = dep_mat.sum(dim=1)
    safe_sum = row_sum.clone()
    safe_sum[safe_sum == 0] = 1.0
    readiness = (dep_mat @ progress_vec) / safe_sum
    readiness[row_sum == 0] = 1.0

    row_sum_t = dep_mat.T.sum(dim=1)
    safe_sum_t = row_sum_t.clone()
    safe_sum_t[safe_sum_t == 0] = 1.0
    validity = (dep_mat.T @ (1.0 - progress_vec)) / safe_sum_t
    validity[row_sum_t == 0] = 1.0

    return readiness, validity


def run_bayes_filter(
    lmm_vsg_scores: torch.Tensor,
    lmm_prog_scores: torch.Tensor,
    dependency_matrix: torch.Tensor,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    T, S = lmm_vsg_scores.shape  # S includes 'None' as last column
    successor_matrix = build_successor_matrix(dependency_matrix)
    static_transition_matrix = successor_matrix / successor_matrix.sum(dim=1, keepdim=True)

    max_vsg_scores = torch.zeros(S - 1)
    monotonic_progress = torch.zeros(S - 1)
    belief = torch.ones(S) / float(S)
    beliefs = []

    for t in range(T):
        vsg_scores_t = lmm_vsg_scores[t]
        max_vsg_scores = torch.maximum(max_vsg_scores, vsg_scores_t[:-1])

        p = lmm_prog_scores[t]
        bins = torch.arange(10, device=p.device, dtype=p.dtype)
        expected_progress = (p * bins).sum(dim=1) / 9.0
        monotonic_progress = torch.maximum(monotonic_progress, expected_progress)

        step_readiness, step_validity = compute_readiness_validity(
            dependency_matrix, monotonic_progress
        )

        transition_t = static_transition_matrix.clone()
        transition_t[:, :-1] = transition_t[:, :-1] * (step_readiness * step_validity)
        transition_t = safe_normalize(transition_t, dim=1)

        # predict
        prior = belief @ transition_t

        # update
        posterior = safe_normalize(prior * vsg_scores_t)

        belief = posterior
        beliefs.append(belief)

    return torch.stack(beliefs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["htstep", "crosstask", "ego4d_goalstep", "coin"],
    )
    parser.add_argument("--model", default="internvl2.5-8b", type=str)
    parser.add_argument("--video_annots_file", required=True, type=str)
    parser.add_argument(
        "--lmm_vsg_dir",
        required=True,
        type=str,
        help="Directory with per-video VSG predictions (torch .pt)",
    )
    parser.add_argument(
        "--lmm_prog_dir",
        type=str,
        required=True,
        help="Directory with per-video progress predictions (torch .pt)",
    )
    parser.add_argument(
        "--llm_prereq_dir",
        type=str,
        required=True,
        help="Directory with per-task prerequisite predictions (torch .pt)",
    )
    parser.add_argument("--output_dir", default="output", type=str)
    return parser.parse_args()


def get_grouping_key(v: Dict[str, Any], dataset: str):
    if dataset == "htstep":
        return (v["activity"], v["variation"])
    if dataset == "crosstask":
        return v["task_id"]
    if dataset == "ego4d_goalstep":
        return v["video_uid"]
    if dataset == "coin":
        return v["activity"]
    raise ValueError(f"Unknown dataset {dataset}")


def key_to_graph_path(key: Any, llm_prereq_dir: str) -> str:
    return os.path.join(llm_prereq_dir, "_".join(key) if isinstance(key, tuple) else str(key))


def get_dependencies(
    video_annots: List[Dict[str, Any]], dataset: str, llm_prereq_dir: str
) -> Dict[Any, Any]:
    groups = defaultdict(list)
    for v in video_annots:
        groups[get_grouping_key(v, dataset)].append(v)

    deps = {}
    for key in groups:
        path = key_to_graph_path(key, llm_prereq_dir)
        deps[key] = torch.load(path + ".pt")
    return deps


def evaluate_video(
    video_annot: Dict[str, Any],
    dataset: str,
    dependencies: torch.Tensor,
    lmm_vsg_dir: str,
    lmm_prog_dir: str,
) -> Dict[str, Any]:
    video_uid = video_annot["video_uid"]
    task_key = get_grouping_key(video_annot, dataset)

    lmm_vsg_scores = torch.load(os.path.join(lmm_vsg_dir, f"{video_uid}.pt"))
    lmm_prog_scores = torch.load(os.path.join(lmm_prog_dir, f"{video_uid}.pt"))

    dependency_matrix = dependencies[task_key]

    beliefs = run_bayes_filter(
        lmm_vsg_scores,
        lmm_prog_scores,
        dependency_matrix,
    )

    return {
        "video_uid": video_uid,
        "task_key": task_key,
        "beliefs": beliefs,
    }


def run_evaluation(
    video_annots: List[Dict[str, Any]],
    args: argparse.Namespace,
    get_Y_pred,
    get_Y_true,
    get_recall,
) -> Dict[str, Any]:
    lmm_vsg_dir = os.path.join(args.lmm_vsg_dir, args.model)
    lmm_prog_dir = os.path.join(args.lmm_prog_dir, args.model)
    dependencies = get_dependencies(video_annots, args.dataset, args.llm_prereq_dir)

    Y_true, Y_pred = {}, {}
    for video_annot in video_annots:
        res = evaluate_video(video_annot, args.dataset, dependencies, lmm_vsg_dir, lmm_prog_dir)
        vid, task_key = res["video_uid"], res["task_key"]
        task = task_key[0] if isinstance(task_key, tuple) else task_key

        y_pred = get_Y_pred(res["beliefs"].numpy())
        y_true = get_Y_true(*y_pred.shape, video_annot)

        Y_pred.setdefault(task, {})[vid] = y_pred
        Y_true.setdefault(task, {})[vid] = y_true

    if args.dataset == "crosstask":
        video_uids = [video_annot["video_uid"] for video_annot in video_annots]

        num_sets = 20
        set_size = 1850
        subset_recalls = []

        for set_idx in range(num_sets):
            random.seed(set_idx)
            subset = random.sample(video_uids, min(set_size, len(video_uids)))
            Y_true_subset = filter_nested_dict_by_video_uids(Y_true, subset)
            Y_pred_subset = filter_nested_dict_by_video_uids(Y_pred, subset)
            recall = get_recall(Y_true_subset, Y_pred_subset)
            recall = np.mean(list(recall.values()))
            subset_recalls.append(recall)

        recall = np.mean(subset_recalls)
    else:
        recall = get_recall(Y_true, Y_pred)

    return recall


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dataset_modules = {
        "htstep": "htstep_recall",
        "crosstask": "crosstask_recall",
        "ego4d_goalstep": "ego4d_goalstep_recall",
        "coin": "coin_recall",
    }
    module = importlib.import_module(dataset_modules[args.dataset])
    get_Y_pred, get_Y_true, get_recall = module.get_Y_pred, module.get_Y_true, module.get_recall

    with open(args.video_annots_file) as f:
        video_annots = json.load(f)

    lmm_vsg_dir = os.path.join(args.lmm_vsg_dir, args.model)
    total_videos = len(video_annots)
    video_annots = [
        v
        for v in video_annots
        if os.path.isfile(os.path.join(lmm_vsg_dir, f"{v['video_uid']}.pt"))
    ]
    print(f"Evaluating {len(video_annots)}/{total_videos} videos in {args.dataset}")

    mean_r1 = run_evaluation(video_annots, args, get_Y_pred, get_Y_true, get_recall) * 100
    print("=" * 50)
    print(f"Mean Recall@1: {mean_r1:.1f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
