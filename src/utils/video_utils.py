import logging
import os

import numpy as np
import torchvision.transforms.functional as F
from torchcodec.decoders import VideoDecoder

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    handler = logging.StreamHandler()  # or FileHandler("your_log_file.log")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def find_video_file(video_dir, vid, extensions=None):
    if extensions is None:
        extensions = [".mp4", ".mkv", ".avi", ".webm"]

    for ext in extensions:
        path = os.path.join(video_dir, f"{vid}{ext}")
        if os.path.exists(path):
            return path
    return None


def get_video_metadata(video_path, device="cuda"):
    try:
        decoder = VideoDecoder(video_path, device=device)
        meta = decoder.metadata

        return {
            "num_frames": meta.num_frames,
            "fps": meta.average_fps,
            "duration": meta.duration_seconds,
        }

    except Exception as e:
        logging.warning(f"Failed to extract metadata for {video_path}: {e}")
        return None


def get_frame_indices(num_frames, fps, sampling_fps):
    num_frames_to_sample = int(num_frames * (sampling_fps / fps))
    frame_indices = np.linspace(0, num_frames - 1, num_frames_to_sample).astype(int)
    return frame_indices.tolist()


def load_video(video_path, frame_indices, device="cuda"):
    # device = "cpu"  # or e.g. "cuda" !
    decoder = VideoDecoder(video_path, device=device)

    # Fallback to CPU if AV1 or VP9 codec is detected
    if decoder.metadata.codec in ("av1", "vp9"):
        logger.warning("Falling back to CPU frame decoding.")
        decoder = VideoDecoder(video_path, device="cpu")

    try:
        return decoder.get_frames_at(indices=frame_indices)
    except RuntimeError as e:
        logger.warning(f"Falling back to manual frame decoding for video '{video_path}': {e}.")
        return safe_load_frames(decoder, frame_indices, video_path)


def safe_load_frames(decoder, frame_indices, video_path):
    frames = []
    target_height, target_width = None, None

    for idx in frame_indices:
        try:
            frame = decoder[idx]

            if target_height is None:
                target_height, target_width = frame.shape[1], frame.shape[2]

            if frame.shape[1:] != (target_height, target_width):
                frame = F.resize(frame, (target_height, target_width))

            frames.append(frame)

        except Exception as e:
            logger.warning(f"Failed to load frame at index {idx} in video '{video_path}': {e}")

    if len(frames) == 0:
        logger.error(f"No frames could be loaded successfully for video: {video_path}")
        raise RuntimeError(f"No frames could be loaded successfully for video: {video_path}")

    return frames
