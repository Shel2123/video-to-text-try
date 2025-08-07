import datetime as dt
import pathlib
import subprocess
import torch
import argparse
from config import (
    DEFAULT_SCENE_THRESHOLD,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_ID
)
from scenedetect import ContentDetector, FrameTimecode, detect
from typing import Tuple


def _sec_to_hhmmss(sec: float) -> str:
    return str(dt.timedelta(seconds=int(sec))).rjust(8, "0")

def _grab_frame(video: str, timestamp: float, out_path: pathlib.Path) -> bool:
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{timestamp:.3f}", "-i", video,
        "-frames:v", "1", str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as err:
        print(f"⚠️  ffmpeg error {err.returncode} — frame skipped")
        return False

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smart scene captions generator (2025 edition)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python3 main.py clip.mp4
  python3 main.py clip.mp4 0.15 --model Salesforce/blip-image-captioning-base""",
    )
    parser.add_argument("video")
    parser.add_argument("scene_thr", nargs="?", type=float, default=DEFAULT_SCENE_THRESHOLD,
                        help="threshold: 0-1 as fraction or 0-100 as %")
    parser.add_argument("max_tokens", nargs="?", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID,
                        help="HF ID of image-to-text model (pipeline)")
    parser.add_argument("--keep-all", action="store_true", help="do not filter identical captions")
    parser.add_argument("--debug", action="store_true", help="verbose log and progress bar")
    return parser.parse_args()

# device select

def _select_device() -> str | int:
    if torch.cuda.is_available():
        print("🖥  CUDA GPU")
        return 0
    if torch.backends.mps.is_available():
        print("🖥  Apple M-series (MPS)")
        return "mps"
    if hasattr(torch, "xpu") and torch.xpu.is_available():  # Intel GPU
        print("🖥  Intel XPU")
        return "xpu"
    print("🖥  CPU")
    return -1

def _detect_scenes(video_path: str, threshold_raw: float, debug: bool) -> list[Tuple[float, float]]:
    # Accept threshold either as fraction (<=1) or integer 0-100
    thr_pct = int(threshold_raw * 100) if threshold_raw <= 1 else int(threshold_raw)
    detector = ContentDetector(threshold=thr_pct)
    scene_list = detect(video_path, detector, show_progress=debug)
    scenes_sec: list[Tuple[float, float]] = [
        (start.get_seconds(), end.get_seconds())  # type: ignore[attr-defined]
        for start, end in scene_list
    ]
    if debug:
        print(f"      Detector returned {len(scenes_sec)} scenes with threshold={thr_pct}")
    return scenes_sec

def _norm(txt: str) -> str:
        return " ".join(txt.casefold().split())