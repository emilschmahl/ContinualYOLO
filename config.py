import torch
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VIDEO_CAPTURE_DEVICE = 1

SCRIPT_DIR = Path(__file__).resolve().parents[1]

SAM2_CHECKPOINT = f"{SCRIPT_DIR}/segment-anything-2-real-time/checkpoints/sam2.1_hiera_small.pt"
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"

# Size must be dividable by 32
FRAME_HEIGHT, FRAME_WIDTH = 640, 640

YOLO_MODEL = "yolo11n.pt"

LEARNING_RATE = 1e-4
SAMPLE_BUFFER_SIZE = 200
SAMPLE_BATCH_SIZE = 7