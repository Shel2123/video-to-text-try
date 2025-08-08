# SceneDescriber

> **Smart scene captions generator (2025 edition)** – transform any video into a concise, live‑style shot‑by‑shot description in seconds.

---

## Key Features

* **Automatic scene detection** using `PySceneDetect`’s content detector
* **Vision‑language captions** powered by Qwen models (tested on [Qwen‑2 VL‑2B‑Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct))
* **Contact‑sheet collage** (optional) for multi‑frame scene context
* Pure‑Python stack,  **CPU‑friendly** , GPU/MPS acceleration when available
* Clean one‑line captions that keep names, actions & a mandatory *shot size* tag
* CLI & Python API, configurable in a single `config.py`
* Apache‑licensed

---

## Table of Contents

1. [Quick start]()
2. [Installation]()
3. [CLI reference]()
4. [Python API]()
5. [Configuration]()
6. [How it works]()
7. [Performance tips]()
8. [Roadmap]()
9. [License]()

---

## Quick start

```bash
# 1. install dependencies (see below for details)
I've used conda
Python ≥ 3.10, Torch ≥ 2.3, Transformers ≥ 4.53, PySceneDetect ≥ 0.6.6.

# 2. run on the demo video
python3 src/main.py data/example.mp4 \
        --model Qwen/Qwen2-VL-2B-Instruct \
        --force-cpu \
        --thr 50 \
        --max-frames 2
```

Example output (truncated):

```
00:00:00 — (Live) In the frame, a castle with a clock tower stands prominently against a backdrop of rolling hills and a lake. The castle is surrounded by lush greenery, and a train is visible in the distance, suggesting a scenic route, wide.
00:00:43 — (Live) The image shows a scenic view of a castle and a town in a valley, with a train passing by in the distance, wide.
00:01:21 — (Live) The frame captures a majestic castle perched on a hill, surrounded by lush greenery and a picturesque village below, wide.
```

---

## Installation

### Prerequisites

| Requirement       | Tested version |
| ----------------- | -------------- |
| **Python**  | 3.10 – 3.12   |
| **PyTorch** | 2.3            |
| **ffmpeg**  | ≥ 4.3        |

SceneDescriber relies on `opencv‑python`, `pillow`, `transformers`, `pyscenedetect` and a few utilities.

> **GPU acceleration** is detected automatically (CUDA, Apple M‑series, Intel XPU). Set `--force-cpu` to bypass.

---

## CLI reference

```text
usage: main.py [-h] [--model MODEL] [--thr THR] [--max-frames N] [--force-cpu] video

positional arguments:
  video                 path to the input video file (any format ffmpeg understands)

optional arguments:
  --model MODEL         Hugging Face ID of the vision‑language model (default: Qwen/…‑2B‑Instruct)
  --thr THR             scene‑change threshold; higher → fewer scenes (default: 30.0)
  --max-frames N        max frames sampled per scene (speed/quality trade‑off)
  --force-cpu           disable any GPU/MPS device and run on CPU only
```

### Common recipes

| Goal                                        | Command                                                            |
| ------------------------------------------- | ------------------------------------------------------------------ |
| Faster captions on a short clip             | `python3 src/main.py clip.mp4 --max-frames 1`                    |
| More detailed captions (risk of repetition) | `python3 src/main.py clip.mp4 --max-frames 6 --thr 20`           |
| High‑end GPU, half‑precision              | `USE_FP16=1 python3 src/main.py clip.mp4 --model <your‑HF‑id>` |

---

## Python API

```python
from src.main import run

lines = run(
    video_path="data/example.mp4",
    model_id="Qwen/Qwen2-VL-2B-Instruct",
    threshold=30.0,
    max_frames_per_scene=3,
    force_cpu=True,
)
print("\n".join(lines))
```

Each element in `lines` is a ready‑to‑print caption starting with a `hh:mm:ss` timestamp.

---

## Configuration

Most knobs live in  **`config.py`** :

| Name                        | Default   | Meaning                                                  |
| --------------------------- | --------- | -------------------------------------------------------- |
| `DEFAULT_SCENE_THRESHOLD` | `0.30`  | Scene detector aggressiveness (0–1 fraction)            |
| `MAX_FRAMES_PER_SCENE`    | `3`     | Hard cap per scene (speed)                               |
| `CONTACT_SHEET`           | `True`  | Create a collage to improve context                      |
| `TILE`                    | `448`   | All frames are resized to `TILE×TILE`before inference |
| `USE_FP16`                | `False` | Run the model in half‑precision if hardware supports it |

Set environment variable **`PYTORCH_ENABLE_MPS_FALLBACK=1`** for smoother Apple Silicon experience (already applied by default).

---

## How it works

1. **Scene detection** – `pyscenedetect` finds hard cuts with a configurable content threshold.
2. **Uniform frame sampling** – up to `N` frames picked per scene to represent motion.
3. **Pre‑processing** – frames are down‑scaled, padded to square and optionally collaged.
4. **Caption generation** – frames + a English prompt are passed to Qwen‑2 VL‑2B; bad words are filtered, temperature/top‑p sampling adds variety.
5. **Live‑style formatting** – captions are prefixed with the scene’s start timestamp and the literal tag  **`(Live)`** .

---

## Performance tips

* **Threshold tuning** – try `--thr 20 … 60`; lower means *more* scenes (finer granularity).
* **Batch processing** – this repo focuses on single‑video simplicity; wrap `run()` in your own loop for large datasets.
* **GPU half‑precision** – export `USE_FP16=1` to cut VRAM usage almost in half.
* **Collage off** – set `CONTACT_SHEET = False` if you prefer random single frames.

---

## Roadmap

* ☑️ Multi‑language prompts
* ☐ Streaming mode (process while recording)
* ☐ Web UI (Gradio)
* ☐ Docker image & CLI installer

---

## License

This project is released under the  **Apache License** .

See LICENSE for details.

---

## Acknowledgements

* [PySceneDetect](https://github.com/Breakthrough/PySceneDetect)
* [Transformers](https://github.com/huggingface/transformers)
* [Qwen‑2 VL](https://huggingface.co/Qwen) from Alibaba & ModelBest
* Inspiration from the amazing open‑source community ❤️

---

> *Made with passion by **[@Shel2123](https://github.com/Shel2123)**– let the machines watch the video so you don’t have to!*
