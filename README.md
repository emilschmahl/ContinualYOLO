# Continual YOLO Live Trainer

A live, camera-driven tool for **continually training a YOLO object detector on the fly**. You click on an object in the camera feed, [SAM2](https://github.com/facebookresearch/segment-anything-2) (real-time fork) segments it, you assign a class name, and the sample is fed straight into a continually-learning YOLO11 model — no separate dataset labeling step required. Once trained, the same window switches into live prediction mode, drawing bounding boxes and an EigenCAM saliency overlay for whatever the model has learned.

## How it works

- **Training mode**: click an object → SAM2 tracks/segments it in real time → type a class name in the console → hit `s` or `r` to save frames as training samples → the model trains incrementally in a background thread, automatically expanding its detection head when a new class appears.
- **Prediction mode**: the trained model runs live inference on the camera feed, drawing labeled bounding boxes and confidence scores.
- Model checkpoints (weights, learned classes, sample buffers) persist to disk, so training picks up where you left off next time.

## Requirements

- Python 3.10+
- A webcam
- (Optional but recommended) an NVIDIA GPU with CUDA for real-time performance. The project also runs on CPU and on Apple Silicon (`mps`), just slower.

## Installation

The project depends on the [`segment-anything-2-real-time`](https://github.com/Gy920/segment-anything-2-real-time) fork of SAM2, which is **not published on PyPI** and must be cloned and linked in manually. Steps 1-2 are the same on every OS; the differences are noted where they come up.

### 1. Clone both repositories

Clone this repository and the SAM2 fork **as sibling folders** (the setup and config expects SAM2 to live one directory above this project):

```bash
mkdir continual-yolo-workspace
cd continual-yolo-workspace

git clone https://github.com/emilschmahl/ContinualYOLO
git clone https://github.com/Gy920/segment-anything-2-real-time.git
```

Your folder structure should look like this:

```
continual-yolo-workspace/
├── ContinualYOLO/                     (this repo)
└── segment-anything-2-real-time/
```

### 2. Create and activate a virtual environment

```bash
cd ContinualYOLO
python -m venv .venv
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

### 3. Install this project's dependencies

```bash
pip install -r requirements.txt
```

### 4. Run `setup.py` to link the SAM2 fork into your environment

```bash
python setup.py
```

### 5. Download the SAM2 checkpoint

Download the checkpoint referenced in `config.py` (`sam2.1_hiera_small.pt` by default) into `segment-anything-2-real-time/checkpoints/`. You can either run the fork's own download script:

**macOS / Linux:**
```bash
cd ../segment-anything-2-real-time/checkpoints
./download_ckpts.sh
```

**Windows:** the script is a bash script; either run it via Git Bash / WSL, or download the checkpoint file manually from the [SAM2 releases page](https://github.com/facebookresearch/segment-anything-2) and place it in that same `checkpoints` folder.

### 6. Configure your camera

Open `config.py` and set `VIDEO_CAPTURE_DEVICE` to your camera's index (`0` is usually the built-in camera, `1` an external one — try a few values if the camera doesn't open).

The camera backend is selected automatically per OS in `main.py` (AVFoundation on macOS, default backend elsewhere), so no changes are needed there.

## Usage

Start the live trainer:

```bash
python main.py
```

A window opens showing the camera feed, starting in **training mode**.

### Training mode

| Action | Effect |
|---|---|
| Left-click an object | Starts SAM2 tracking/segmentation on that object |
| Type a name + Enter (in the console) | Assigns a class label to the selected object |
| `s` | Saves the current frame + mask as a training sample |
| `r` | Toggles continuous recording (saves a sample every ~0.1s while held on) |
| `m` | Toggles the segmentation mask overlay |
| `b` | Toggles the bounding box overlay |
| `e` | Switches to prediction mode |
| `q` | Stops training, saves the model, and quits |

Each saved sample is queued and trained on in a background thread, so the camera feed keeps running smoothly while the model learns. New class names automatically grow the model's detection head; only the newly added output layer is trained, while the rest of the network stays frozen (i.e. lightweight continual fine-tuning rather than training from scratch).

### Prediction mode

| Action | Effect |
|---|---|
| `e` | Switches back to training mode |
| `x` | Toggles an EigenCAM saliency overlay, showing which regions the model is focusing on |
| `q` | Quits |

In this mode the model runs live inference on the camera feed and draws labeled, confidence-scored bounding boxes for everything it currently recognizes.

### Persistence

On quit, the model's weights, learned class list, and per-class sample buffers are saved to `continual_yolo.pt` (configurable via `CONTINUAL_MODEL` in `config.py`). Restarting `main.py` automatically loads this checkpoint and continues from where you left off; if no checkpoint exists yet, a fresh model is initialized from the pretrained YOLO11 weights.

## Configuration reference (`config.py`)

| Setting | Description |
|---|---|
| `DEVICE` | Auto-detected: CUDA → MPS (Apple Silicon) → CPU |
| `VIDEO_CAPTURE_DEVICE` | Camera index |
| `SAM2_CHECKPOINT` / `MODEL_CONFIG` | Path to the SAM2 checkpoint and its model config |
| `FRAME_HEIGHT` / `FRAME_WIDTH` | Capture resolution, center-cropped; must be divisible by 32 |
| `YOLO_MODEL` | Base pretrained YOLO checkpoint to start from |
| `LEARNING_RATE` | Optimizer learning rate for continual training |
| `SAMPLE_BUFFER_SIZE` | Max samples kept per class for replay during training |
| `SAMPLE_BATCH_SIZE` | Samples drawn per class per training step |
| `CONFIDENCE_EMA` / `EIGENCAM_EMA` | Smoothing factors to reduce frame-to-frame jitter |
| `CONF_THRESHOLD` / `IOU_THRESHOLD` / `MAX_DETECTIONS` | Non-max suppression settings for prediction |

## Troubleshooting

- **`ImportError: cannot import name 'build_sam2_camera_predictor'`** — usually means a *different* `sam2` package (e.g. the official non-real-time PyPI/GitHub version) already exists in your venv's `site-packages` and is shadowing the fork. Delete that folder (and its matching `.dist-info` folder if present) and re-verify with the check command in step 4.
- **`OSError: CUDA_HOME environment variable is not set`** — only relevant if you try to `pip install -e .` inside the SAM2 fork directly; the `.pth`-file method in step 4 avoids this entirely.
- **No camera image on Windows** — make sure `main.py` isn't forcing `cv2.CAP_AVFOUNDATION` (that's macOS-only); it should auto-select the right backend per OS.