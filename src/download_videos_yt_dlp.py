"""Download COIN dataset videos from YouTube using yt-dlp.

This script reads a COIN dataset JSON file and downloads the corresponding
videos into subfolders grouped by `recipe_type`.

Example:
    python download_coin_videos.py \
        --output_dir /path/to/data/COIN/videos \
        --json_path ./COIN.json \
        --cookies ./cookies.txt \
        --subset testing \
        --max_workers 4
"""

import argparse
import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import yt_dlp


def download_video(youtube_id: str, info: dict, output_dir: str, cookies_path: str) -> str:
    """Download a single video from YouTube with separate audio/video merging."""
    recipe_type = info["recipe_type"]
    video_dir = os.path.join(output_dir, str(recipe_type))
    os.makedirs(video_dir, exist_ok=True)

    failed_log = os.path.join(output_dir, "failed_downloads.txt")
    video_url = f"https://www.youtube.com/watch?v={youtube_id}"

    # Skip if video already downloaded
    if glob.glob(os.path.join(video_dir, f"{youtube_id}.*")):
        return f"✔️ Already downloaded: {youtube_id}"

    def _log_failed(log_file: str, vid: str):
        """Append failed video ID to a text file."""
        with open(log_file, "a") as f:
            f.write(f"{vid}\n")

    def format_selector(ctx):
        """Select best separate video/audio formats if available."""
        formats = ctx.get("formats", [])[::-1]
        formats = [
            f for f in formats if all(k in f for k in ("vcodec", "acodec", "ext", "protocol"))
        ]

        # Best video-only stream
        best_video = next(
            (f for f in formats if f["vcodec"] != "none" and f["acodec"] == "none"), None
        )
        if not best_video:
            _log_failed(failed_log, youtube_id)
            return

        # Match audio extension
        audio_ext = {"mp4": "m4a", "webm": "webm"}.get(best_video["ext"])
        best_audio = next(
            (
                f
                for f in formats
                if f["vcodec"] == "none" and f["acodec"] != "none" and f["ext"] == audio_ext
            ),
            None,
        )

        if best_audio:
            yield {
                "format_id": f"{best_video['format_id']}+{best_audio['format_id']}",
                "ext": best_video["ext"],
                "requested_formats": [best_video, best_audio],
                "protocol": f"{best_video['protocol']}+{best_audio['protocol']}",
            }
        else:
            # Fallback: combined audio+video stream
            best_fallback = next(
                (f for f in formats if f["vcodec"] != "none" and f["acodec"] != "none"), None
            )
            if best_fallback:
                yield best_fallback
            else:
                _log_failed(failed_log, youtube_id)
                return

    ydl_opts = {
        "format": format_selector,
        "outtmpl": os.path.join(video_dir, f"{youtube_id}.%(ext)s"),
        "quiet": False,
        "cookies": os.path.abspath(cookies_path),
        # Alternatively: 'cookies_from_browser': ('chrome',),
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        return f"✅ Downloaded: {youtube_id}"
    except yt_dlp.utils.DownloadError:
        _log_failed(failed_log, youtube_id)
        return f"❌ Failed: {youtube_id}"


def parse_args():
    parser = argparse.ArgumentParser(description="Download COIN dataset videos using yt-dlp.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(os.getcwd(), "COIN/videos"),
        help="Directory where videos will be saved (default: ./COIN/videos)",
    )
    parser.add_argument(
        "--json_path", type=str, default="./COIN.json", help="Path to the COIN dataset JSON file."
    )
    parser.add_argument(
        "--cookies", type=str, default="./cookies.txt", help="Path to cookies.txt file (optional)."
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="testing",
        choices=["training", "validation", "testing"],
        help="Subset to download (default: testing)",
    )
    parser.add_argument(
        "--max_workers", type=int, default=1, help="Number of parallel downloads (default: 1)"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.json_path) as f:
        data = json.load(f)["database"]

    videos = {k: v for k, v in data.items() if v["subset"] == args.subset}

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(download_video, vid_id, info, args.output_dir, args.cookies)
            for vid_id, info in videos.items()
        ]
        for future in as_completed(futures):
            print(future.result())
