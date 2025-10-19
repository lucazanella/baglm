import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from typing import OrderedDict

import torch
from tqdm import tqdm

import t2v_metrics
from utils.text_utils import canonicalize_step


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
    parser.add_argument("--result_dir", default="./results/ego4d_goalstep", type=str)
    parser.add_argument("--annot_dir", type=str, help="Directory containing the annotations")
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--oai_key_path", default="./_OAI_KEY.txt", type=str)
    return parser.parse_args()


def process_video(annot, args, score_func, kwargs, result_dir):
    video_uid = annot["video_uid"]
    start_time, end_time = annot["start_time"], annot["end_time"]
    goal_clip_uid = f"{video_uid}_{start_time}_{end_time}"
    goal_description = annot["goal_description"]

    clip_annots = [
        {
            "clip_uid": video_uid,
            "video_start_sec": start_time,
            "video_end_sec": end_time,
            "annotations": [{"language_queries": None, "annotation_uid": goal_clip_uid}],
        }
    ]

    original_to_canonical = OrderedDict()
    language_queries = []
    for segment in annot["segments"]:
        original = segment["step_description"]
        canonical = canonicalize_step(original)
        if original not in original_to_canonical:
            original_to_canonical[original] = canonical
        query = {
            "clip_start_sec": segment["start_time"],
            "clip_end_sec": segment["end_time"],
            "query": canonical,
            "query_original": original,
        }
        language_queries.append(query)

    clip_annots[0]["annotations"][0]["language_queries"] = language_queries
    raw_texts = [str(query["query"]) for query in language_queries]
    raw_texts = list(dict.fromkeys(raw_texts))

    save_path = os.path.join(result_dir, f"{video_uid}.pt")
    if os.path.exists(save_path):
        print(f"Result file {save_path} already exists. Skipping.")
        return video_uid

    batch = [
        {"goal": goal_description, "step_a": a, "step_b": b}
        for a, b in product(raw_texts, repeat=2)
    ]

    try:
        scores = score_func.batch_forward_text(batch, args.batch_size, **kwargs).cpu()
        torch.save(scores, save_path)
        return video_uid
    except Exception as e:
        print(f"Failed processing {video_uid}: {e}")
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

    json_fl = f"{args.annot_dir}/goalstep_val.json"
    annotations = json.load(open(json_fl))["videos"]
    annotations = [annot for annot in annotations if annot["video_uid"]]
    if args.num_workers == 0:
        # Run sequentially
        for annot in tqdm(annotations, total=len(annotations)):
            process_video(annot, args, score_func, kwargs, result_dir)
    else:
        # Run in parallel using threads
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [
                executor.submit(process_video, annot, args, score_func, kwargs, result_dir)
                for annot in annotations
            ]
            for future in tqdm(as_completed(futures), total=len(futures)):
                future.result()


if __name__ == "__main__":
    main()
