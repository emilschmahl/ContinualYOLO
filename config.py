import torch
from pathlib import Path

DEVICE = "cuda"

VIDEO_CAPTURE_DEVICE = 0

CONTINUAL_MODEL = Path(__file__).resolve().parent / "continual_yolo.pt"

SCRIPT_DIR = Path(__file__).resolve().parent

SAM2_CHECKPOINT = f"{SCRIPT_DIR}/segment-anything-2-real-time/checkpoints/sam2.1_hiera_small.pt"
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"

# Size must be dividable by 32
FRAME_HEIGHT, FRAME_WIDTH = 640, 640

YOLO_MODEL = "yolo11n.pt"

LEARNING_RATE = 1e-4
SAMPLE_BUFFER_SIZE = 500
SAMPLE_BATCH_SIZE = 7

CONFIDENCE_EMA = 0.6

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 20