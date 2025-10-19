import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

import pandas as pd
import torch
from tqdm import tqdm

import t2v_metrics


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
    parser.add_argument("--result_dir", default="./results/coin", type=str)
    parser.add_argument("--annotations_json", type=str, default="data/annotations.json")
    parser.add_argument("--taxonomy_csv", type=str, default="data/taxonomy.csv")
    parser.add_argument("--split", type=str, default="testing", choices=("testing",))
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--oai_key_path", default="./_OAI_KEY.txt", type=str)
    return parser.parse_args()


def process_activity(activity, steps, args, score_func, kwargs, result_dir):
    save_path = os.path.join(result_dir, f"{activity}.pt")
    batch = [{"goal": activity, "step_a": a, "step_b": b} for a, b in product(steps, repeat=2)]

    if os.path.exists(save_path):
        print(f"Result file {save_path} already exists. Skipping.")
        return activity

    try:
        scores = score_func.batch_forward_text(batch, args.batch_size, **kwargs).cpu()
        torch.save(scores, save_path)
        return activity
    except Exception as e:
        print(f"Failed processing {activity}: {e}")
        return None


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

    taxonomy = pd.read_excel(args.taxonomy_csv, sheet_name="target_action_mapping")

    act2steps = {
        target_label: list(group["Action Label"].values)
        for target_label, group in taxonomy.groupby("Target Label", sort=False)
    }

    if args.num_workers == 0:
        # Run sequentially
        for activity, steps in tqdm(act2steps.items(), total=len(act2steps)):
            process_activity(activity, steps, args, score_func, kwargs, result_dir)
    else:
        # Run in parallel using threads
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [
                executor.submit(
                    process_activity,
                    activity,
                    steps,
                    args,
                    score_func,
                    kwargs,
                    result_dir,
                )
                for activity, steps in act2steps.items()
            ]
            for future in tqdm(as_completed(futures), total=len(futures)):
                future.result()


if __name__ == "__main__":
    main()
