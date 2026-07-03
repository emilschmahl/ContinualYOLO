import torch
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VIDEO_CAPTURE_DEVICE = 1

DEST_DIR = Path(__file__).resolve().parent / "samples"
SCRIPT_DIR = Path(__file__).resolve().parents[1]

SAM2_CHECKPOINT = f"{SCRIPT_DIR}/segment-anything-2-real-time/checkpoints/sam2.1_hiera_small.pt"
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"