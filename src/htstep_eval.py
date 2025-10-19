import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import t2v_metrics
from constants import PROMPT_PROGRESS, PROMPT_VSG
from dataset import HTStep


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, help="Directory containing the dataset")
    parser.add_argument(
        "--root_dir", default="./datasets", type=str, help="Root directory for saving datasets."
    )
    parser.add_argument("--cache_dir", default=t2v_metrics.constants.HF_CACHE_DIR, type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--visual_batch_size", default=1, type=int)
    parser.add_argument("--text_batch_size", default=32, type=int)
    parser.add_argument("--model", default="internvl2.5-8b", type=str)
    parser.add_argument("--question", default=None, type=str)
    parser.add_argument("--question_file", default=None, type=str)
    parser.add_argument("--answer", default=None, type=str)
    parser.add_argument("--result_dir", default="./results/htstep", type=str)
    parser.add_argument(
        "--video_annots_file", default="datasets/htstep/htstep_video_annots.json", type=str
    )
    parser.add_argument("--annotations_json", type=str, default="data/annotations.json")
    parser.add_argument("--taxonomy_csv", type=str, default="data/taxonomy.csv")
    parser.add_argument("--split", type=str, default="val_seen", choices=("val_seen",))
    parser.add_argument("--segment_duration", type=int, default=1)
    parser.add_argument("--sampling_fps", type=int, default=2)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=1000000)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--prompt_type", choices=[PROMPT_VSG, PROMPT_PROGRESS], required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.root_dir, exist_ok=True)
    result_dir = os.path.join(args.result_dir, args.model)
    os.makedirs(result_dir, exist_ok=True)

    print(f"Performance of {args.model}.")
    score_func = t2v_metrics.get_score_model(
        model=args.model, device=args.device, cache_dir=args.cache_dir
    )
    preprocess_fn = score_func.model.get_preprocessor()

    dataset = HTStep(
        dataset_dir=args.dataset_dir,
        annotations_json=args.annotations_json,
        taxonomy_csv=args.taxonomy_csv,
        split=args.split,
        result_dir=result_dir,
        root_dir=args.root_dir,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        preprocess_fn=preprocess_fn,
        segment_duration=args.segment_duration,
        sampling_fps=args.sampling_fps,
        video_annots_file=args.video_annots_file,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda x: x,
    )
    assert args.batch_size == 1, "batch_size must be 1 for custom batch_forward"

    kwargs = {}
    if args.question_file is not None:
        with open(args.question_file) as f:
            args.question = f.read()
    if args.question is not None:
        print(f"Using question template: {args.question}")
        kwargs["question_template"] = args.question
    if args.answer is not None:
        print(f"Using answer template: {args.answer}")
        kwargs["answer_template"] = args.answer

    for batch_data in tqdm(dataloader, total=len(dataloader)):
        for video_data in batch_data:
            video_uid = video_data["video_uid"]
            result_path = os.path.join(result_dir, f"{video_uid}.pt")

            if os.path.exists(result_path):
                print(f"Result file {result_path} already exists. Skipping.")
                scores = torch.load(result_path)
            else:
                if args.prompt_type == PROMPT_VSG:
                    scores = score_func.forward_vsg(
                        video_data,
                        args.visual_batch_size,
                        **kwargs,
                    ).cpu()
                elif args.prompt_type == PROMPT_PROGRESS:
                    scores = score_func.forward_progress(
                        video_data,
                        args.visual_batch_size,
                        args.text_batch_size,
                        **kwargs,
                    ).cpu()
                else:
                    raise ValueError(f"Unknown prompt_type: {args.prompt_type}")

                scores = scores.repeat_interleave(args.segment_duration, dim=0)
                torch.save(scores, result_path)


if __name__ == "__main__":
    main()
