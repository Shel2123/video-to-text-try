# main.py
import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2  # pip install opencv-python
import torch as _torch
from PIL import Image
from scenedetect import detect, ContentDetector  # pip install scenedetect
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from transformers.utils.logging import set_verbosity_error

set_verbosity_error()

# ---------- config ----------
MAX_FRAMES_PER_SCENE = 3       # limit for speed
CONTACT_SHEET = True           # collage
TILE = 448                     # base small size
USE_FP16 = True               # fp16 on GPU/MPS
PATCH_MULT = 32               # mod for ViT-patch (Qwen2-VL)

# Recomend of MPS
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


# ---------- utils ----------
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

def detect_scenes(video_path: str, threshold: float = 30.0) -> List[Scene]:
    """
    threshold — for ContentDetector (average 27–45). Try by yourself.
    """
    pairs = detect(video_path, ContentDetector(threshold=int(threshold)), show_progress=True)
    scenes: List[Scene] = [Scene(s.get_seconds(), e.get_seconds()) for s, e in pairs]
    if not scenes:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        cap.release()
        dur = frames / (fps if fps > 1e-6 else 1.0)
        scenes = [Scene(0.0, dur)]
    return scenes

def sample_times_uniform(start: float, end: float, max_frames: int = 16) -> List[float]:
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

def pad_to_multiple(im: Image.Image, m: int = PATCH_MULT) -> Image.Image:
    """Padding m (ViT-патчей), to avoid MPS fall."""
    w, h = im.size
    nw = ((w + m - 1) // m) * m
    nh = ((h + m - 1) // m) * m
    if (nw, nh) == (w, h):
        return im
    canvas = Image.new("RGB", (nw, nh), (0, 0, 0))
    canvas.paste(im, (0, 0))
    return canvas

def make_contact_sheet(frames: list[Image.Image], cols=4, tile=TILE) -> Image.Image:
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
    return pad_to_multiple(sheet, PATCH_MULT)

def to_square_canvas(im: Image.Image, size: int = TILE, mult: int = PATCH_MULT) -> Image.Image:
    """
    Letterbox to exactly size×size (по центру) and padding for division `mult`.
    Fixed normalixzation = lower killed MPSGraph on matrix.
    """
    w, h = im.size
    if w == 0 or h == 0:
        return Image.new("RGB", (size, size), (0, 0, 0))
    scale = size / max(w, h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    im = im.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(im, ((size - nw) // 2, (size - nh) // 2))
    return pad_to_multiple(canvas, mult)


# ---------- VLM wrapper ----------
class Describer:
    def __init__(self, model_id: str):
        # выбор девайса
        # if _torch.backends.mps.is_available():
        #     device = "mps"
        # elif _torch.cuda.is_available():
        #     device = "cuda"
        # else:
        device = "cpu"

        dtype = _torch.float16 if device != "cpu" and USE_FP16 else _torch.float32

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self.model.to(device)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
        self.device = device

    def describe_scene(self, frames: list[Image.Image], scene_ts: str) -> str:
        if not frames:
            return f"{scene_ts} — (Лайф) Нет визуальной информации, общий"

        frames = [to_square_canvas(downscale(f, short=TILE), size=TILE, mult=PATCH_MULT) for f in frames]
        if CONTACT_SHEET and len(frames) > 1:
            frames = [to_square_canvas(make_contact_sheet(frames, cols=4, tile=TILE), size=TILE, mult=PATCH_MULT)]

        messages = [{
            "role": "user",
            "content": [*([{"type": "image"}] * len(frames)),
                        {"type": "text", "text":
                            f"Верни РОВНО ОДНУ строку:\n"
                            f"{scene_ts} — (Лайф) <краткое описание>, <общий/средний/крупный>\n"
                            f"Без лишних слов."
                         }],
        }]

        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(
            text=prompt, images=frames,
            return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        try:
            gen_kwargs = dict(
                max_new_tokens=48, 
                do_sample=False, 
                use_cache=False,
            )
            allowed = {
                "input_ids",
                "attention_mask",
                "pixel_values",
                "image_grid_thw",
            }
            gen_inputs = {k: v for k, v in inputs.items() if k in allowed}
            with _torch.inference_mode():
                out_ids = self.model.generate(**gen_inputs, **gen_kwargs)
        except RuntimeError as e:
            # fallback CPU
            if "MPS" in self.device:
                self.model = self.model.to("cpu").to(dtype=_torch.float32)
                inputs = {k: v.to("cpu") if hasattr(v, "to") else v for k, v in inputs.items()}
                with _torch.inference_mode():
                    out_ids = self.model.generate(**gen_inputs, max_new_tokens=48, do_sample=False)
            else:
                raise e

        text = self.processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip().splitlines()[0]

        # clear
        del inputs, out_ids, frames
        if self.device == "mps":
            _torch.mps.empty_cache()

        return text if text.startswith(scene_ts) else f"{scene_ts} — (Лайф) {text}"

# ---------- main flow ----------
def run(video_path: str,
        model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
        threshold: float = 30.0,
        max_frames_per_scene: int = 16) -> List[str]:
    scenes = detect_scenes(video_path, threshold=threshold)
    desc = Describer(model_id)

    lines: List[str] = []
    for sc in scenes:
        ts = hhmmss(sc.start_sec)
        times = sample_times_uniform(sc.start_sec, sc.end_sec, max_frames=max_frames_per_scene)
        frames: List[Image.Image] = []
        frames = [
            im for t in times
            if (im := grab_frame(video_path, t)) is not None
        ]
        if not frames:
            lines.append(f"{ts} — (Лайф) Нет визуальной информации, общий")
            continue
        # safety: batch size
        if len(frames) > max_frames_per_scene:
            step = math.ceil(len(frames) / max_frames_per_scene)
            frames = frames[::step][:max_frames_per_scene]

        line = desc.describe_scene(frames, ts)
        if not line.startswith(ts):
            line = f"{ts} — (Лайф) {line}"
        print(line, flush=True)
        lines.append(line)
        
    return lines

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=str)
    ap.add_argument("--model", type=str,
                    default="Qwen/Qwen2-VL-7B-Instruct",
                    help="Напр.: Qwen/Qwen2-VL-7B-Instruct или llava-hf/llava-onevision-qwen2-7b-ov-hf")
    ap.add_argument("--thr", type=float, default=30.0, help="trashhold ContentDetector (PySceneDetect)")
    ap.add_argument("--max-frames", type=int, default=3, help="Max F/SC")
    args = ap.parse_args()

    lines = run(args.video, model_id=args.model, threshold=args.thr, max_frames_per_scene=args.max_frames)
    print("\n".join(lines))

if __name__ == "__main__":
    main()
