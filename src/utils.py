import math
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
from scenedetect import ContentDetector, detect
from typing import Tuple
from PIL import Image
import cv2
from dataclasses import dataclass
from scenedetect import detect, ContentDetector
from config import TILE


# LATEST ===============================

@dataclass
class Scene:
    start_sec: float
    end_sec: float

def hhmmss(t: float) -> str:
    t = max(t, 0.0)
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def detect_scenes(video_path: str, threshold: float = 30.0) -> list[Scene]:
    """
    threshold — for ContentDetector (average 27–45). Try by yourself.
    """
    pairs = detect(video_path, ContentDetector(threshold=int(threshold)), show_progress=True)
    scenes: list[Scene] = [Scene(s.get_seconds(), e.get_seconds()) for s, e in pairs]
    if not scenes:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        cap.release()
        dur = frames / (fps if fps > 1e-6 else 1.0)
        scenes = [Scene(0.0, dur)]
    return scenes

def sample_times_uniform(start: float, end: float, max_frames: int = 16) -> list[float]:
    if end <= start:
        return [start]
    # hard 2 
    n = min(max_frames, max(1, int((end - start) * 2)))
    if n == 1:
        # middle
        return [start + (end - start) * 0.5]
    return [start + (end - start) * i / (n - 1) for i in range(n)]

def grab_frame(video_path: str, t_sec: float) -> Image.Image | None:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)

def downscale(im: Image.Image, short=TILE) -> Image.Image:
    w, h = im.size
    if min(w, h) <= short:
        return im
    if w < h:
        nw, nh = short, int(h * short / w)
    else:
        nw, nh = int(w * short / h), short
    return im.resize((nw, nh), Image.BICUBIC)

def make_contact_sheet(frames: list[Image.Image], cols=3, tile=TILE) -> Image.Image:
    fr = [downscale(f, short=tile) for f in frames[:cols * 2]]
    if not fr:
        return Image.new("RGB", (tile, tile), (0, 0, 0))
    rows = math.ceil(len(fr) / cols)
    tw = max(f.width for f in fr)
    th = max(f.height for f in fr)
    sheet = Image.new("RGB", (tw * cols, th * rows), (0, 0, 0))
    for i, f in enumerate(fr):
        r, c = divmod(i, cols)
        x = c * tw + (tw - f.width) // 2
        y = r * th + (th - f.height) // 2
        sheet.paste(f, (x, y))
    return sheet

def to_square_canvas(im: Image.Image, size: int = TILE) -> Image.Image:
    # no padding
    w, h = im.size
    if w == 0 or h == 0:
        return Image.new("RGB", (size, size), (0, 0, 0))
    scale = size / max(w, h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    im = im.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(im, ((size - nw) // 2, (size - nh) // 2))
    return canvas






# LEGACY ===============================

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
