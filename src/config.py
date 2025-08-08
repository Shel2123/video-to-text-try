import os

# latest
DEFAULT_SCENE_THRESHOLD = 0.30  # 30 %
MAX_FRAMES_PER_SCENE = 3       # limit for speed
CONTACT_SHEET = True           # collage
TILE = 448                     # base small size
USE_FP16 = False               # fp16 on GPU/MPS
PATCH_MULT = 14               # mod for ViT-patch (Qwen2-VL)


# LEGACY =======================================
DEFAULT_STEP_SEC = 5     # fallback sampling step
DEFAULT_MAX_TOKENS = 40
DEFAULT_MODEL_ID = "nlpconnect/vit-gpt2-image-captioning"
FRAME_EXT = "png"
OFFLINE = os.getenv("VSC_OFFLINE") == "1"
