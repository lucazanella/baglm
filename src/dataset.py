import collections
import json
import os

import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.crosstask_utils import get_language_queries, get_vids, read_task_info
from utils.text_utils import canonicalize_step
from utils.video_utils import (
    find_video_file,
    get_frame_indices,
    get_video_metadata,
    load_video,
)


def remove_duplicate_annotations(ants, tol=1e-3):
    # remove duplicate / very short annotations (same category and starting/ending time)
    valid_events = []
    for event in ants:
        if "label_id" not in event:
            event["label_id"] = str(int(event["id"]) - 1)
        s, e, l = event["segment"][0], event["segment"][1], event["label_id"]  # noqa: E741
        if (e - s) >= tol:
            valid = True
        else:
            valid = False
        for p_event in valid_events:
            if (
                (abs(s - p_event["segment"][0]) <= tol)
                and (abs(e - p_event["segment"][1]) <= tol)
                and (l == p_event["label_id"])
            ):
                valid = False
                break
        if valid:
            valid_events.append(event)
    return valid_events


class Ego4DGoalStep(Dataset):
    def __init__(
        self,
        annot_dir,
        video_dir,
        result_dir,
        preprocess_fn=None,
        root_dir="./",
        start_idx=0,
        end_idx=1000000,
        segment_duration=1,
        sampling_fps=2,
        video_annots_file=None,
    ):
        self.annot_dir = annot_dir
        self.video_dir = video_dir
        self.dataset_name = "ego4d_goalstep"
        self.root_dir = os.path.join(root_dir, self.dataset_name)
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir, exist_ok=True)

        self.preprocess_fn = preprocess_fn

        self.segment_duration = segment_duration
        self.sampling_fps = sampling_fps

        ego4d_goalstep_video_to_original_to_canonical_file = os.path.join(
            self.root_dir, "ego4d_goalstep_video_to_original_to_canonical.json"
        )
        ego4d_goalstep_video_annots_file = video_annots_file or os.path.join(
            self.root_dir, "ego4d_goalstep_video_annots.json"
        )
        if os.path.exists(ego4d_goalstep_video_annots_file):
            with open(ego4d_goalstep_video_annots_file) as f:
                self.video_annots = json.load(f)
            self.video_annots = self.video_annots[start_idx:end_idx]
            self.video_annots = [
                vid
                for vid in self.video_annots
                if not os.path.exists(os.path.join(result_dir, f"{vid['video_uid']}.pt"))
            ]
            return

        json_fl = f"{self.annot_dir}/goalstep_val.json"
        annotations = json.load(open(json_fl))["videos"]

        stats = collections.Counter()

        extensions = [".mp4"]
        failed_videos = []
        vid_metadata = {}

        for annot in annotations:
            video_uid = annot["video_uid"]
            video_path = find_video_file(self.video_dir, video_uid, extensions)
            metadata = get_video_metadata(video_path)
            if metadata is None:
                failed_videos.append(video_uid)
                continue

            metadata["ext"] = os.path.splitext(video_path)[-1]
            vid_metadata[video_uid] = metadata

        annotations = [annot for annot in annotations if annot["video_uid"] in vid_metadata]

        self.video_to_original_to_canonical = {}
        self.video_annots = []
        for annot in annotations:
            video_uid = annot["video_uid"]
            stats["videos"] += 1

            start_time, end_time = annot["start_time"], annot["end_time"]
            goal_clip_uid = f"{video_uid}_{start_time}_{end_time}"
            goal_category = annot["goal_category"]
            goal_description = annot["goal_description"]

            clip_annots = [
                {
                    "clip_uid": video_uid,  # full video = 1 clip
                    "video_start_sec": start_time,
                    "video_end_sec": end_time,
                    "annotations": [
                        {"language_queries": None, "annotation_uid": goal_clip_uid}
                    ],  # fill in language queries
                }
            ]

            original_to_canonical = collections.OrderedDict()
            language_queries = []
            for segment in annot["segments"]:
                original = segment["step_description"]
                canonical = canonicalize_step(original)
                if original not in original_to_canonical:
                    original_to_canonical[original] = canonical
                assert len(canonical) > 0, "len(canonical) == 0"

                query = {
                    "clip_start_sec": segment["start_time"],  # video == clip
                    "clip_end_sec": segment["end_time"],  # video == clip
                    "query": canonical,
                    "query_original": original,
                }

                assert (
                    segment["end_time"] - segment["start_time"] > 0
                ), "end_time - start_time <= 0"
                language_queries.append(query)
                stats["query"] += 1
                stats["step"] += 1

            self.video_to_original_to_canonical[video_uid] = original_to_canonical

            seen = set()
            step_headline = []
            for canonical in original_to_canonical.values():
                if canonical not in seen:
                    step_headline.append(canonical)
                    seen.add(canonical)

            clip_annots[0]["annotations"][0]["language_queries"] = language_queries
            self.video_annots.append(
                {
                    "video_uid": video_uid,
                    "video_ext": vid_metadata[video_uid]["ext"],
                    "video_num_frames": vid_metadata[video_uid]["num_frames"],
                    "video_fps": vid_metadata[video_uid]["fps"],
                    "video_duration": vid_metadata[video_uid]["duration"],
                    "goal_category": goal_category,
                    "goal_description": goal_description,
                    "step_headline": step_headline,
                    "clips": clip_annots,
                }
            )

        json.dump(self.video_annots, open(ego4d_goalstep_video_annots_file, "w"))
        # json.dump(self.video_to_original_to_canonical, open(ego4d_goalstep_video_to_original_to_canonical_file, 'w'))
        with open(ego4d_goalstep_video_to_original_to_canonical_file, "w") as f:
            json.dump(self.video_to_original_to_canonical, f, indent=4)

    def __len__(self):
        return len(self.video_annots)

    def __getitem__(self, idx):
        item = self.video_annots[idx]

        video_uid = item["video_uid"]
        video_path = os.path.join(self.video_dir, f"{video_uid}{item['video_ext']}")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file for {video_uid} not found.")

        num_frames, fps = item["video_num_frames"], item["video_fps"]

        frame_indices = get_frame_indices(num_frames, fps, self.sampling_fps)

        all_frames = load_video(video_path, frame_indices)
        frames_per_segment = int(self.segment_duration * self.sampling_fps)
        kwargs = {
            "nframes": frames_per_segment,
        }
        if self.preprocess_fn is not None:
            all_frames = [self.preprocess_fn(frame.data, **kwargs) for frame in all_frames]

        num_frames = len(all_frames)
        video_segments = []
        for i in range(0, num_frames, frames_per_segment):
            segment_frames = all_frames[i : i + frames_per_segment]
            if len(segment_frames) == frames_per_segment:
                video_segments.append(torch.cat(segment_frames))

        clips = item["clips"]
        raw_texts = [
            str(query["query"]) for query in clips[0]["annotations"][0]["language_queries"]
        ]
        raw_texts = list(dict.fromkeys(raw_texts))
        assert (
            raw_texts == item["step_headline"]
        ), f"raw_texts: {raw_texts} != item['step_headline']: {item['step_headline']}"
        texts = [
            {"goal": item["goal_description"], "step": step} for step in item["step_headline"]
        ]

        return {
            "video_uid": video_uid,
            "task_id": item["goal_category"],
            "videos": video_segments,
            "texts": texts,
        }


class CrossTask(Dataset):
    def __init__(
        self,
        dataset_dir,
        result_dir,
        preprocess_fn=None,
        root_dir="./",
        use_related=False,
        start_idx=0,
        end_idx=1000000,
        segment_duration=1,
        sampling_fps=2,
        video_annots_file=None,
    ):
        self.dataset_dir = dataset_dir
        self.root_dir = os.path.join(root_dir, "crosstask")
        os.makedirs(self.root_dir, exist_ok=True)

        self.preprocess_fn = preprocess_fn

        self.segment_duration = segment_duration
        self.sampling_fps = sampling_fps

        # self.video_dir = os.path.join(dataset_dir, "videos_val")
        self.video_dir = os.path.join(dataset_dir, "videos")
        # self.video_csv_path = os.path.join(dataset_dir, "crosstask_release/videos_val.csv")
        self.video_csv_path = os.path.join(dataset_dir, "crosstask_release/videos.csv")
        self.annot_dir = os.path.join(dataset_dir, "crosstask_release/annotations")

        primary_path = os.path.join(dataset_dir, "crosstask_release/tasks_primary.txt")
        related_path = os.path.join(dataset_dir, "crosstask_release/tasks_related.txt")

        crosstask_video_annots_file = video_annots_file or os.path.join(
            self.root_dir, "crosstask_video_annots.json"
        )
        if os.path.exists(crosstask_video_annots_file):
            with open(crosstask_video_annots_file) as f:
                self.video_annots = json.load(f)
            self.video_annots = self.video_annots[start_idx:end_idx]
            self.video_annots = [
                vid
                for vid in self.video_annots
                if not os.path.exists(os.path.join(result_dir, f"{vid['video_uid']}.pt"))
            ]
            return

        val_vids = get_vids(self.video_csv_path)
        primary_info = read_task_info(primary_path)

        if use_related:
            related_info = read_task_info(related_path)
            self.task_steps = {**primary_info["steps"], **related_info["steps"]}
            n_steps = {**primary_info["n_steps"], **related_info["n_steps"]}
        else:
            self.task_steps = primary_info["steps"]
            n_steps = primary_info["n_steps"]

        all_tasks = set(n_steps.keys())
        val_vids = {task: vids for task, vids in val_vids.items() if task in all_tasks}
        extensions = [".mp4", ".webm"]
        failed_videos = []
        vid_metadata = {}

        for task, vids in val_vids.items():
            for vid in vids:
                video_path = find_video_file(self.video_dir, vid, extensions)
                if not video_path:
                    continue

                metadata = get_video_metadata(video_path)
                if metadata is None:
                    failed_videos.append(vid)
                    continue

                metadata["ext"] = os.path.splitext(video_path)[-1]
                vid_metadata[vid] = metadata

        val_vids = {
            task: [vid for vid in vids if vid in vid_metadata] for task, vids in val_vids.items()
        }

        self.video_annots = [
            {
                "video_uid": vid,
                "video_ext": vid_metadata[vid]["ext"],
                "video_num_frames": vid_metadata[vid]["num_frames"],
                "video_fps": vid_metadata[vid]["fps"],
                "video_duration": vid_metadata[vid]["duration"],
                "task_id": task,
                "task_name": primary_info["title"][task],
                "task_steps": self.task_steps[task],
                "clips": [
                    {
                        "clip_uid": vid,
                        "annotations": [
                            {
                                "language_queries": get_language_queries(
                                    os.path.join(self.annot_dir, f"{task}_{vid}.csv"),
                                    self.task_steps[task],
                                ),
                                "annotation_uid": f"{task}_{vid}",
                            }
                        ],
                    }
                ],
            }
            for task, vids in val_vids.items()
            for vid in vids
        ]

        with open(crosstask_video_annots_file, "w") as f:
            json.dump(self.video_annots, f)

    def __len__(self):
        return len(self.video_annots)

    def __getitem__(self, idx):
        item = self.video_annots[idx]

        video_uid = item["video_uid"]
        video_path = os.path.join(self.video_dir, f"{video_uid}{item['video_ext']}")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file for {video_uid} not found.")

        num_frames, fps = item["video_num_frames"], item["video_fps"]

        frame_indices = get_frame_indices(num_frames, fps, self.sampling_fps)

        all_frames = load_video(video_path, frame_indices)
        frames_per_segment = int(self.segment_duration * self.sampling_fps)
        kwargs = {
            "nframes": frames_per_segment,
        }
        if self.preprocess_fn is not None:
            all_frames = [self.preprocess_fn(frame.data, **kwargs) for frame in all_frames]

        num_frames = len(all_frames)
        video_segments = []
        for i in range(0, num_frames, frames_per_segment):
            segment_frames = all_frames[i : i + frames_per_segment]
            if len(segment_frames) == frames_per_segment:
                video_segments.append(torch.cat(segment_frames))

        texts = [{"goal": item["task_name"], "step": step} for step in item["task_steps"]]

        return {
            "video_uid": video_uid,
            "videos": video_segments,
            "task_id": item["task_id"],
            "texts": texts,
        }


class HTStep(Dataset):
    def __init__(
        self,
        dataset_dir,
        annotations_json,
        taxonomy_csv,
        split,
        result_dir,
        preprocess_fn=None,
        root_dir="./",
        start_idx=0,
        end_idx=1000000,
        segment_duration=1,
        sampling_fps=2,
        video_annots_file=None,
    ):
        self.dataset_dir = dataset_dir
        self.split = split
        self.root_dir = os.path.join(root_dir, "htstep")
        os.makedirs(self.root_dir, exist_ok=True)

        self.preprocess_fn = preprocess_fn
        self.video_dir = os.path.join(dataset_dir, f"videos/{split}")

        self.segment_duration = segment_duration
        self.sampling_fps = sampling_fps

        taxonomy = pd.read_csv(taxonomy_csv)
        annotations = json.load(open(annotations_json))

        htstep_video_annots_file = video_annots_file or os.path.join(
            self.root_dir, "htstep_video_annots.json"
        )
        if os.path.exists(htstep_video_annots_file):
            with open(htstep_video_annots_file) as f:
                self.video_annots = json.load(f)
            self.video_annots = self.video_annots[start_idx:end_idx]
            self.video_annots = [
                vid
                for vid in self.video_annots
                if not os.path.exists(os.path.join(result_dir, f"{vid['video_uid']}.pt"))
            ]
            return

        ants = {kk: vv for kk, vv in annotations.items() if vv["subset"] == split}

        print(f"{len(ants)} annotations found for {split} split")

        activity2step_ids = {
            kk: vv.global_step_index.values for kk, vv in taxonomy.groupby("activity")
        }
        activity2idx = {
            kk: ii
            for ii, (kk, vv) in enumerate(sorted(activity2step_ids.items(), key=lambda x: x[1][0]))
        }
        step_idx2headline = taxonomy.set_index("global_step_index")["headline"].to_dict()

        # -- for every annotated video get the variation and make a list of steps in that variation
        stepid2variation = {row.global_step_index: row.variation for _, row in taxonomy.iterrows()}
        vid2var = {}
        for vid, vid_annotations in ants.items():
            var = None
            for seg in vid_annotations["annotations"]:
                if var is None:
                    var = stepid2variation[seg["id"]]
                assert var == stepid2variation[seg["id"]]
            vid2var[vid] = var

        act2var2steps = {
            kk: {kkk: set(vvv.global_step_index.values) for kkk, vvv in vv.groupby("variation")}
            for kk, vv in taxonomy.groupby("activity")
        }

        extensions = [".mp4", ".webm"]
        failed_videos = []
        vid_metadata = {}

        for vid in ants.keys():
            video_path = find_video_file(self.video_dir, vid, extensions)
            if not video_path:
                continue

            metadata = get_video_metadata(video_path)
            if metadata is None:
                failed_videos.append(vid)
                continue

            metadata["ext"] = os.path.splitext(video_path)[-1]
            vid_metadata[vid] = metadata

        ants = {
            vid: vid_annotations for vid, vid_annotations in ants.items() if vid in vid_metadata
        }

        self.video_annots = [
            {
                "video_uid": vid,
                "video_ext": vid_metadata[vid]["ext"],
                "video_num_frames": vid_metadata[vid]["num_frames"],
                "video_fps": vid_metadata[vid]["fps"],
                "video_duration": vid_annotations["duration"],
                "activity_idx": activity2idx[vid_annotations["activity"]],
                "activity": vid_annotations["activity"],
                "variation": vid_annotations["variation"],
                "step_headline": [
                    step_idx2headline[step_idx]
                    for step_idx in act2var2steps[ants[vid]["activity"]][ants[vid]["variation"]]
                ],
                "clips": [
                    {
                        "annotations": [
                            {
                                "language_queries": [
                                    {
                                        "clip_start_sec": float(event["segment"][0]),
                                        "clip_end_sec": float(event["segment"][1]),
                                        "query": step_idx2headline[event["id"]],
                                        "step": event["id"],
                                        "partial": event["partial"],
                                    }
                                    for event in remove_duplicate_annotations(
                                        vid_annotations["annotations"]
                                    )
                                ]
                            }
                        ]
                    }
                ],
            }
            for vid, vid_annotations in ants.items()
        ]

        with open(htstep_video_annots_file, "w") as f:
            json.dump(self.video_annots, f)

    def __len__(self):
        return len(self.video_annots)

    @property
    def video_lengths(self):
        length_list = []
        for sample in self.video_annots:
            length_list.append(sample["video_end_sec"])
        return length_list

    def __getitem__(self, idx):
        item = self.video_annots[idx]

        video_uid = item["video_uid"]
        video_path = os.path.join(self.video_dir, f"{video_uid}{item['video_ext']}")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file for {video_uid} not found.")

        num_frames, fps = item["video_num_frames"], item["video_fps"]

        frame_indices = get_frame_indices(num_frames, fps, self.sampling_fps)

        all_frames = load_video(video_path, frame_indices)
        frames_per_segment = int(self.segment_duration * self.sampling_fps)
        kwargs = {
            "nframes": frames_per_segment,
        }
        if self.preprocess_fn is not None:
            all_frames = [self.preprocess_fn(frame.data, **kwargs) for frame in all_frames]

        num_frames = len(all_frames)
        video_segments = []
        for i in range(0, num_frames, frames_per_segment):
            segment_frames = all_frames[i : i + frames_per_segment]
            if len(segment_frames) == frames_per_segment:
                video_segments.append(torch.cat(segment_frames))

        texts = [{"goal": item["activity"], "step": step} for step in item["step_headline"]]

        return {
            "video_uid": video_uid,
            "videos": video_segments,
            "task_id": item["activity_idx"],
            "texts": texts,
        }


class COIN(Dataset):
    def __init__(
        self,
        dataset_dir,
        annotations_json,
        taxonomy_csv,
        split,
        result_dir,
        preprocess_fn=None,
        root_dir="./",
        start_idx=0,
        end_idx=1000000,
        segment_duration=1,
        sampling_fps=2,
        video_annots_file=None,
    ):
        self.dataset_dir = dataset_dir
        self.split = split
        self.root_dir = os.path.join(root_dir, "coin")
        os.makedirs(self.root_dir, exist_ok=True)

        self.preprocess_fn = preprocess_fn
        self.video_dir = os.path.join(dataset_dir, "videos")

        self.segment_duration = segment_duration
        self.sampling_fps = sampling_fps

        taxonomy = pd.read_excel(taxonomy_csv, sheet_name="target_action_mapping")
        annotations = json.load(open(annotations_json))

        coin_video_annots_file = video_annots_file or os.path.join(
            self.root_dir, "coin_video_annots.json"
        )
        if os.path.exists(coin_video_annots_file):
            with open(coin_video_annots_file) as f:
                self.video_annots = json.load(f)
            self.video_annots = self.video_annots[start_idx:end_idx]
            self.video_annots = [
                vid
                for vid in self.video_annots
                if not os.path.exists(os.path.join(result_dir, f"{vid['video_uid']}.pt"))
            ]
            return

        ants = {kk: vv for kk, vv in annotations["database"].items() if vv["subset"] == split}
        act2steps = {
            target_label: list(group["Action Label"].values)
            for target_label, group in taxonomy.groupby("Target Label", sort=False)
        }

        print(f"{len(ants)} annotations found for {split} split")

        extensions = [".mp4", ".webm"]
        failed_videos = []
        vid_metadata = {}

        for vid in ants.keys():
            recipe_type = str(ants[vid]["recipe_type"])
            video_dir = os.path.join(self.video_dir, recipe_type)
            video_path = find_video_file(video_dir, vid, extensions)
            if not video_path:
                continue

            metadata = get_video_metadata(video_path)
            if metadata is None:
                failed_videos.append(vid)
                continue

            metadata["ext"] = os.path.splitext(video_path)[-1]
            vid_metadata[vid] = metadata

        ants = {
            vid: vid_annotations for vid, vid_annotations in ants.items() if vid in vid_metadata
        }

        self.video_annots = [
            {
                "video_uid": vid,
                "video_ext": vid_metadata[vid]["ext"],
                "video_num_frames": vid_metadata[vid]["num_frames"],
                "video_fps": vid_metadata[vid]["fps"],
                "video_duration": vid_annotations["duration"],
                "activity": vid_annotations["class"],
                "activity_idx": vid_annotations["recipe_type"],
                # "step_headline": [
                #     event["label"] for event in vid_annotations["annotation"]
                # ],
                "step_headline": act2steps[vid_annotations["class"]],
                "clips": [
                    {
                        "annotations": [
                            {
                                "language_queries": [
                                    {
                                        "clip_start_sec": float(event["segment"][0]),
                                        "clip_end_sec": float(event["segment"][1]),
                                        "query": event["label"],
                                        "step": event["id"],
                                        "partial": False,
                                    }
                                    for event in vid_annotations["annotation"]
                                ]
                            }
                        ]
                    }
                ],
            }
            for vid, vid_annotations in ants.items()
        ]

        with open(coin_video_annots_file, "w") as f:
            json.dump(self.video_annots, f)

    def __len__(self):
        return len(self.video_annots)

    @property
    def video_lengths(self):
        length_list = []
        for sample in self.video_annots:
            length_list.append(sample["video_end_sec"])
        return length_list

    def __getitem__(self, idx):
        item = self.video_annots[idx]

        video_uid = item["video_uid"]
        video_dir = os.path.join(self.video_dir, str(item["activity_idx"]))
        video_path = os.path.join(video_dir, f"{video_uid}{item['video_ext']}")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file for {video_uid} not found.")

        num_frames, fps = item["video_num_frames"], item["video_fps"]

        frame_indices = get_frame_indices(num_frames, fps, self.sampling_fps)

        all_frames = load_video(video_path, frame_indices)
        frames_per_segment = int(self.segment_duration * self.sampling_fps)
        kwargs = {
            "nframes": frames_per_segment,
        }
        if self.preprocess_fn is not None:
            all_frames = [self.preprocess_fn(frame.data, **kwargs) for frame in all_frames]

        num_frames = len(all_frames)
        video_segments = []
        for i in range(0, num_frames, frames_per_segment):
            segment_frames = all_frames[i : i + frames_per_segment]
            if len(segment_frames) == frames_per_segment:
                video_segments.append(torch.cat(segment_frames))

        texts = [{"goal": item["activity"], "step": step} for step in item["step_headline"]]

        return {
            "video_uid": video_uid,
            "videos": video_segments,
            "task_id": item["activity_idx"],
            "texts": texts,
        }
