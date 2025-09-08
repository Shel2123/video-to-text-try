import os
import sys
import math
import argparse
from contextlib import nullcontext
from typing import Optional
import logging
import inspect
from types import SimpleNamespace

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch as _torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.logging import set_verbosity_error
import transformers
import scenedetect

logger = logging.getLogger("fastvlm")

from config import (
    MAX_FRAMES_PER_SCENE,
    TILE,
    CONTACT_SHEET,
    USE_FP16,
    DEFAULT_SCENE_THRESHOLD,
    DEFAULT_MODEL_ID,
    DEFAULT_MAX_TOKENS,
)
from utils import (
    to_square_canvas,
    downscale,
    make_contact_sheet,
    detect_scenes,
    hhmmss,
    grab_frame,
    sample_times_uniform,
    open_frame_grabber
)

set_verbosity_error()

def _log_runtime_banner(device: _torch.device, dtype: _torch.dtype):
    logger.debug(
        f"[runtime] device={device.type}  dtype={str(dtype).split('.')[-1]}  "
        f"mps_available={_torch.backends.mps.is_available()}  "
        f"mps_built={_torch.backends.mps.is_built()}",
    )

def _get_vision_processor(model) -> object:
    """
    Возвращает image_processor модели (у разных реализаций имя отличается).
    """
    for attr_chain in [
        ("get_vision_tower", "image_processor"),
        ("get_vision_tower", "vision_tower", "image_processor"),
        ("vision_tower", "image_processor"),
        ("visual", "image_processor"),
        ("get_vision_module", "image_processor"),
    ]:
        try:
            obj = model
            for attr in attr_chain:
                obj = getattr(obj, attr) if isinstance(attr, str) else obj[attr]
                if callable(obj) and attr.startswith("get_"):
                    obj = obj()
            if obj is not None:
                return obj
        except Exception:
            pass
    raise RuntimeError("Не удалось получить image_processor у модели (vision tower).")

class Describer:
    """
    Описание сцены через Apple FastVLM (trust_remote_code).
    """

    def __init__(self, model_id: str, force_cpu: bool = False, num_threads: Optional[int] = None):
        if force_cpu:
            self._device = _torch.device("cpu")
        elif _torch.cuda.is_available():
            self._device = _torch.device("cuda")
        elif _torch.backends.mps.is_available():
            self._device = _torch.device("mps")
        else:
            self._device = _torch.device("cpu")

        if isinstance(num_threads, int) and num_threads > 0:
            try:
                _torch.set_num_threads(num_threads)
            except Exception:
                pass

        model_dtype = _torch.float16 if (self._device.type == "cuda" and USE_FP16) else _torch.float32

        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if getattr(self.tok, "pad_token_id", None) is None and getattr(self.tok, "eos_token_id", None) is not None:
            try:
                self.tok.pad_token = self.tok.eos_token
            except Exception:
                self.tok.pad_token_id = int(self.tok.eos_token_id)

        load_kwargs = dict(
            trust_remote_code=True,
            torch_dtype=model_dtype,
        )
        if self._device.type == "cuda":
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = None

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        if load_kwargs.get("device_map") != "auto":
            self.model.to(self._device)
        self.model.eval()

        try:
            cfg = getattr(self.model, "config", None)
            if cfg is not None:
                if hasattr(cfg, "attn_implementation"):
                    cfg.attn_implementation = "eager"
                elif hasattr(cfg, "_attn_implementation"):
                    setattr(cfg, "_attn_implementation", "eager")
        except Exception:
            pass

        self._eos_token_ids = self._collect_eos_ids()

        self.IMAGE_TOKEN_INDEX = getattr(getattr(self.model, "config", object()), "image_token_index", -200)

        apply_sdpa_patch = os.getenv("FASTVLM_PATCH_SDPA", "0") == "1"
        if apply_sdpa_patch and self._device.type == "mps":
            try:
                import torch.nn.functional as F
                _orig_sdpa = F.scaled_dot_product_attention
                def _sdpa_safe(q,k,v,*args,**kwargs):
                    try:
                        Hq, Hk = q.shape[-3], k.shape[-3]
                        if Hq != Hk and Hk > 0 and (Hq % Hk == 0) and (Hq//Hk) <= 2:
                            rep = Hq // Hk
                            k = k.repeat_interleave(rep, dim=-3).contiguous()
                            v = v.repeat_interleave(rep, dim=-3).contiguous()
                    except Exception:
                        pass
                    return _orig_sdpa(q,k,v,*args,**kwargs)
                F.scaled_dot_product_attention = _sdpa_safe
                logger.warning("[patch] SDPA(GQA) on MPS enabled (FASTVLM_PATCH_SDPA=1).")
            except Exception as _e:
                logger.warning(f"[patch] SDPA monkey-patch skipped: {_e}")


        _log_runtime_banner(self._device, model_dtype)

    def _collect_eos_ids(self) -> Optional[list[int]]:
        ids = set()
        for obj in (getattr(self, "model", None) and getattr(self.model, "generation_config", None), self.tok):
            if not obj:
                continue
            for name in ("eos_token_id", "eot_token_id", "im_end_id", "eom_token_id", "end_of_text_token_id"):
                val = getattr(obj, name, None)
                if isinstance(val, int):
                    ids.add(int(val))
                elif isinstance(val, (list, tuple)):
                    ids.update(int(x) for x in val if isinstance(x, int))
        for tok in ("<|eot_id|>", "<|im_end|>", "<|endoftext|>"):
            try:
                tid = self.tok.convert_tokens_to_ids(tok)
                if isinstance(tid, int) and tid != getattr(self.tok, "unk_token_id", -1):
                    ids.add(int(tid))
            except Exception:
                pass
        return sorted(ids) if ids else None

    def _get_model_device(self) -> _torch.device:
        return getattr(self.model, "device", self._device)

    def _prepare_image_tensor(self, pil_image: Image.Image) -> _torch.Tensor:
        proc = _get_vision_processor(self.model)
        px = proc(images=pil_image, return_tensors="pt")["pixel_values"]
        dev = self._get_model_device()
        model_dtype = getattr(self.model, "dtype", _torch.float32)
        return px.to(dev, dtype=model_dtype).contiguous()
    
    def _encode_with_image(self, messages: list[dict], pil_image: Image.Image):
        """
        Строго по рекомендации Apple FastVLM:
        1) рендерим шаблон в строку (tokenize=False),
        2) разделяем по '<image>',
        3) токенизируем куски без спец. токенов,
        4) вставляем IMAGE_TOKEN_INDEX (-200) между частями.
        """
        rendered = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        occ = rendered.count("<image>")
        if occ != 1:
            raise RuntimeError(f"Expected exactly one <image> placeholder, got {occ}.")

        if "<image>" not in rendered:
            raise RuntimeError("В prompt отсутствует плейсхолдер <image>.")
        pre, post = rendered.split("<image>", 1)
        pre_ids = self.tok(pre,  return_tensors="pt", add_special_tokens=False).input_ids
        post_ids = self.tok(post, return_tensors="pt", add_special_tokens=False).input_ids
        img_tok = _torch.tensor([[int(self.IMAGE_TOKEN_INDEX)]], dtype=pre_ids.dtype)
        input_ids = _torch.cat([pre_ids, img_tok, post_ids], dim=1)
        attention_mask = _torch.ones_like(input_ids)
        dev = self._get_model_device()
        input_ids = input_ids.to(dev)
        attention_mask = attention_mask.to(dev)
        px = self._prepare_image_tensor(pil_image)
        try:
            cnt = int((input_ids == int(self.IMAGE_TOKEN_INDEX)).sum().item())
            logger.debug(f"[tokens] IMAGE_TOKEN_INDEX occurrences: {cnt}")
        except Exception:
            pass
        return input_ids, attention_mask, px

    def describe_scene(self, frames: list[Image.Image], scene_ts: str) -> str:
        if not frames:
            return f"{scene_ts} — (Live) No visual information."

        frames = [to_square_canvas(downscale(f, short=TILE), size=TILE) for f in frames]
        if CONTACT_SHEET and len(frames) > 1:
            frames = [to_square_canvas(make_contact_sheet(frames, cols=3, tile=TILE), size=TILE)]
        img = frames[0].convert("RGB")

        instruction = "Describe super briefly the scene focusing on salient visual elements. Avoid preambles. At the end, add which shot: long, medium, or close-up."
        system_prompt = "Always answer in English. Two concise sentences (total 15-30 words), no quotes."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "<image>\n" + instruction},
        ]

        input_ids, attention_mask, px = self._encode_with_image(messages, img)

        gen_kwargs = dict(
            max_new_tokens=min(DEFAULT_MAX_TOKENS, 60),
            penalty_alpha=0.6,
            top_k=4,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            pad_token_id=int(self.tok.pad_token_id) if getattr(self.tok, "pad_token_id", None) is not None else None,
            eos_token_id=self._eos_token_ids,
        )

        sdpa_ctx = nullcontext()
        try:
            from torch.nn.attention import sdpa_kernel, SDPBackend
            dev_type = self._get_model_device().type
            if dev_type in {"mps", "cpu", "cuda"}:
                sdpa_ctx = sdpa_kernel(SDPBackend.MATH)
                logger.debug(f"[sdpa] using MATH backend on {dev_type}")
        except Exception as e:
            logger.debug(f"[sdpa] using no-op ctx: {e}")

        def _generate_with(px_tensor, **kw):
            try:
                return self.model.generate(
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    images=px_tensor,
                    **kw,
                )
            except TypeError as e:
                if "images" in str(e) and "unexpected" in str(e):
                    return self.model.generate(
                        inputs=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=px_tensor,
                        **kw,
                    )
                raise

        def _decode(greedy_out):
            seq = getattr(greedy_out, "sequences", greedy_out)
            if not isinstance(seq, _torch.Tensor):
                seq = _torch.as_tensor(seq)

            prefix_len = int(input_ids.shape[1])
            ref_ids = input_ids.detach()
            if ref_ids.device != seq.device:
                ref_ids = ref_ids.to(seq.device)

            if seq.shape[1] > prefix_len and _torch.equal(seq[0, :prefix_len], ref_ids[0]):
                new_tokens = seq[0, prefix_len:]
                decoded = self.tok.decode(new_tokens, skip_special_tokens=True)
            else:
                decoded = self.tok.decode(seq[0], skip_special_tokens=True)

            text = (decoded or "").strip()
            text = text.strip(' "\'“”«»')
            text = (decoded or "").strip()
            text = text.strip(' "\'“”«»')
            return text  # без split по \n


        def _looks_degenerate(s: str) -> bool:
            import re
            low = s.lower()
            if len(low) < 8:
                return False
            if re.search(r'(\b\w{1,3}\b)(\s+\1){6,}', low):
                return True
            if re.search(r'(.)\1{8,}', low):
                return True
            return False

        with _torch.inference_mode():
            with sdpa_ctx:
                try:
                    logger.debug(f"[gen] dev={input_ids.device} img_dev={px.device} img_dtype={px.dtype}")
                    out = _generate_with(px, **gen_kwargs)
                except RuntimeError as e:
                    if self._device.type == "mps" and ("mps" in str(e).lower() or "MPS" in str(e)):
                        logger.warning("[warn] MPS kernel failed; retrying on CPU once...")
                        self.model.to("cpu")
                        input_ids = input_ids.to("cpu")
                        attention_mask = attention_mask.to("cpu")
                        px = px.to("cpu")
                        logger.debug(f"[gen-retry-cpu] dev={input_ids.device} img_dev={px.device} img_dtype={px.dtype}")
                        out = _generate_with(px, **gen_kwargs)
                    else:
                        raise

        text = _decode(out)
        if (not text) or _looks_degenerate(text):
            relax = dict(
                max_new_tokens=min(DEFAULT_MAX_TOKENS, 60),
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                no_repeat_ngram_size=6,
                use_cache=True,
                return_dict_in_generate=True,
                pad_token_id=int(self.tok.pad_token_id) if getattr(self.tok, "pad_token_id", None) is not None else None,
                eos_token_id=self._eos_token_ids,
            )
            with _torch.inference_mode():
                with sdpa_ctx:
                    out = _generate_with(px, **relax)
            text = _decode(out)

        # на крайний случай — формально соблюдаем формат
        if not text:
            text = "содержимое не распознано"
        return f"{scene_ts} — (Live) {text}"

def run(
    video_path: str,
    model_id: str = DEFAULT_MODEL_ID,
    threshold: float = DEFAULT_SCENE_THRESHOLD,
    max_frames_per_scene: int = MAX_FRAMES_PER_SCENE,
    force_cpu: bool = False,
) -> list[str]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Видео не найдено: {video_path}")

    scenes = detect_scenes(video_path, threshold=threshold)
    if not scenes:
        logger.warning("Предупреждение: сцены не найдены, будет описан весь ролик целиком.")
    desc = Describer(model_id, force_cpu=force_cpu)

    lines: list[str] = []
    with open_frame_grabber(video_path) as fb:
        if not scenes:
            duration = getattr(fb, "duration", None)
            if duration is None:
                frame_count = getattr(fb, "frame_count", None)
                fps = getattr(fb, "fps", None)
                duration = float(frame_count) / float(fps) if frame_count and fps else 0.0
            scenes = [SimpleNamespace(start_sec=0.0, end_sec=float(duration))]
        for sc in scenes:
            ts = hhmmss(sc.start_sec)
            times = sample_times_uniform(sc.start_sec, sc.end_sec, max_frames=max_frames_per_scene)
            frames = [im for t in times if (im := fb.grab_at(t)) is not None]
            if len(frames) > max_frames_per_scene:
                step = math.ceil(len(frames) / max_frames_per_scene)
                frames = frames[::step][:max_frames_per_scene]

            if not frames:
                line = f"{ts} — (Live) No visual information."
            else:
                if CONTACT_SHEET and len(frames) > 1:
                    sheet = make_contact_sheet(
                        [to_square_canvas(downscale(f, short=TILE), size=TILE) for f in frames],
                        cols=3, tile=TILE
                    )
                    img = to_square_canvas(sheet, size=TILE).convert("RGB")
                else:
                    img = to_square_canvas(downscale(frames[0], short=TILE), size=TILE).convert("RGB")
                line = desc.describe_scene([img], ts)


            print(line, flush=True)
            lines.append(line)

    return lines

def main(logger: logging.Logger):
    ap = argparse.ArgumentParser(
        description="Описывает сцены в видео с помощью мультимодели (например, apple/FastVLM-1.5B)."
    )
    ap.add_argument("video", type=str, help="Путь к видеофайлу (например, data/example.mp4)")
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL_ID, help=f"ID модели (по умолчанию: {DEFAULT_MODEL_ID})")
    ap.add_argument("--threshold", type=float, default=DEFAULT_SCENE_THRESHOLD,
                    help=f"Порог детектора сцен (целое в диапазоне ~27–45, по умолчанию {DEFAULT_SCENE_THRESHOLD})")
    ap.add_argument("--max-frames", type=int, default=MAX_FRAMES_PER_SCENE,
                    help=f"Макс. кадров на сцену (по умолчанию {MAX_FRAMES_PER_SCENE})")
    ap.add_argument("--cpu", action="store_true", dest="force_cpu", help="Принудительно использовать CPU.")
    ap.add_argument("-v", "--verbose", action="store_true", help="Подробный лог (DEBUG).")
    args: argparse.Namespace = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, 
        format='[%(levelname)s] %(message)s',
        stream=sys.stderr,
    )
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.debug(f"[versions] torch={_torch.__version__}")
    logger.debug(f"[versions] transformers={transformers.__version__}  scenedetect={scenedetect.__version__}")

    try:
        run(
            video_path=args.video,
            model_id=args.model,
            threshold=args.threshold,
            max_frames_per_scene=args.max_frames,
            force_cpu=args.force_cpu,
        )
    except FileNotFoundError as e:
        logger.error(f"Ошибка: {e}")
        sys.exit(2)
    except RuntimeError as e:
        logger.error(f"RuntimeError: {e}")
        if _torch.backends.mps.is_available() and not args.force_cpu:
            logger.warning("Совет: запустите с флагом --cpu, чтобы проверить, связано ли с MPS.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Непредвиденная ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main(logger)
