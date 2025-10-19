import argparse
import os
from itertools import product

import torch
from tqdm import tqdm

import t2v_metrics
from utils.crosstask_utils import get_vids, read_task_info


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, help="Directory containing the dataset")
    parser.add_argument(
        "--root_dir", default="./datasets", type=str, help="Root directory for saving datasets."
    )
    parser.add_argument("--cache_dir", default=t2v_metrics.constants.HF_CACHE_DIR, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--model", default="llama-3.3-70b-instruct", type=str)
    parser.add_argument("--question_file", default=None, type=str)
    parser.add_argument("--system_file", default=None, type=str)
    parser.add_argument("--question", default=None, type=str)
    parser.add_argument("--answer", default=None, type=str)
    parser.add_argument("--result_dir", default="./results/crosstask", type=str)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--oai_key_path", default="./_OAI_KEY.txt", type=str)
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.root_dir, exist_ok=True)
    result_dir = os.path.join(args.result_dir, args.model)
    os.makedirs(result_dir, exist_ok=True)

    print(f"Performance of {args.model}.")
    if "gpt" in args.model:
        with open(args.oai_key_path) as f:
            key = f.read().strip()
        score_func = t2v_metrics.get_score_model(model=args.model, api_key=key)
    else:
        score_func = t2v_metrics.get_score_model(model=args.model)

    kwargs = {}
    if args.system_file is not None:
        with open(args.system_file) as f:
            kwargs["system_prompt"] = f.read()
    if args.question_file is not None:
        with open(args.question_file) as f:
            args.question = f.read()
    if args.question is not None:
        print(f"Using question template: {args.question}")
        kwargs["question_template"] = args.question
    if args.answer is not None:
        print(f"Using answer template: {args.answer}")
        kwargs["answer_template"] = args.answer

    use_related = False
    video_csv_path = os.path.join(args.dataset_dir, "crosstask_release/videos.csv")

    primary_path = os.path.join(args.dataset_dir, "crosstask_release/tasks_primary.txt")
    related_path = os.path.join(args.dataset_dir, "crosstask_release/tasks_related.txt")

    val_vids = get_vids(video_csv_path)
    primary_info = read_task_info(primary_path)

    if use_related:
        related_info = read_task_info(related_path)
        task_steps = {**primary_info["steps"], **related_info["steps"]}
        n_steps = {**primary_info["n_steps"], **related_info["n_steps"]}
    else:
        task_steps = primary_info["steps"]
        n_steps = primary_info["n_steps"]

    all_tasks = set(n_steps.keys())
    val_vids = {task: vids for task, vids in val_vids.items() if task in all_tasks}

    for task, vids in tqdm(val_vids.items(), total=len(val_vids)):
        steps = task_steps[task]
        save_path = os.path.join(result_dir, f"{task}.pt")
        batch = [{"goal": task, "step_a": a, "step_b": b} for a, b in product(steps, repeat=2)]

        if os.path.exists(save_path):
            print(f"Result file {save_path} already exists. Skipping.")
            scores = torch.load(save_path)
        else:
            scores = score_func.batch_forward_text(batch, args.batch_size, **kwargs).cpu()
            torch.save(scores, save_path)


if __name__ == "__main__":
    main()
