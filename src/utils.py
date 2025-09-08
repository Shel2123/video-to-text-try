import math
from dataclasses import dataclass
from typing import Tuple
import contextlib

import cv2
from PIL import Image
from scenedetect import ContentDetector, detect


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
    thr = int(round(threshold))
    thr = max(0, min(255, thr))
    pairs = detect(video_path, ContentDetector(threshold=thr), show_progress=True)
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
    n = min(max_frames, max(1, int((end - start) * 2)))
    if n == 1:
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


def downscale(im: Image.Image, short: int) -> Image.Image:
    w, h = im.size
    if min(w, h) <= short:
        return im
    if w < h:
        nw, nh = short, int(h * short / w)
    else:
        nw, nh = int(w * short / h), short
    return im.resize((nw, nh), Image.Resampling.BICUBIC)


def make_contact_sheet(frames: list[Image.Image], cols: int, tile: int) -> Image.Image:
    fr = [downscale(f, short=tile) for f in frames[: cols * 2]]
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


def to_square_canvas(im: Image.Image, size: int) -> Image.Image:
    w, h = im.size
    if w == 0 or h == 0:
        return Image.new("RGB", (size, size), (0, 0, 0))
    scale = size / max(w, h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    im = im.resize((nw, nh), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(im, ((size - nw) // 2, (size - nh) // 2))
    return canvas

class FrameGrabber:
    def __init__(self, video_path: str):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Видео не открывается: {video_path}")

    def grab_at(self, t_sec: float) -> Image.Image | None:
        self.cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_sec) * 1000.0)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame)

    def close(self):
        self.cap.release()

@contextlib.contextmanager
def open_frame_grabber(video_path: str):
    fb = FrameGrabber(video_path)
    try:
        yield fb
    finally:
        fb.close()
