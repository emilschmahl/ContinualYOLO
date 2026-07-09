# Continual YOLO Live Trainer

Train a YOLO object detector live with your camera. Choose an object by clicking on it, assign a class name and create training data automatically. The object is segmented by SAM2 and  fed into a continually-learning YOLO11 model live. The model is expanded on-the-fly if you add a new class. Switch to the prediction mode any time to see your model in action. 

## How it works

- **Training mode**: 
  - click on an object  
  - assign a class name in the console 
  - hit `s` (save one sample) or `r` (record) to save frames as training samples 
  - the model trains incrementally in a background thread
- **Prediction mode**: the trained model runs live inference on the camera feed, drawing labeled bounding boxes and confidence scores. Press `x` to toggle the EigenCAM overlay
- Model checkpoints (weights, learned classes, sample buffers) persist to disk, so training picks up where you left off next time.

## Requirements

- Python 3.10+
- NVIDIA GPU (necessary for PyTorch CUDA)
- A webcam

## Installation

### 1. Clone this repository

```bash
git clone --recursive https://github.com/emilschmahl/ContinualYOLO
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

### 3. Run `setup.py`

```bash
python setup.py
```

### 4. (Optional) Configure your camera (and other settings)

Open `config.py` 
```bash
python config.py
```
and set `VIDEO_CAPTURE_DEVICE` to your camera's index (`0` is usually the built-in camera, `1` an external one).

## Usage

Start the live trainer:

```bash
python continual_training.py
```

A window opens showing the camera feed, starting in **training mode**.

### Training mode

| Action        | Effect                                                                  |
|---------------|-------------------------------------------------------------------------|
| Left-click    | Starts SAM2 tracking/segmentation on that object                        |
| console input | Assigns a class label to the selected object                            |
| `s`           | Train YOLO on current frame                                             |
| `r`           | Toggles continuous recording (saves a sample every ~0.1s while held on) |
| `m`           | Toggles the segmentation mask overlay                                   |
| `b`           | Toggles the bounding box overlay                                        |
| `e`           | Switches to prediction mode                                             |
| `q`           | Stops training, saves the model, and quits                              |


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

| Setting | Description                                                 |
|---|-------------------------------------------------------------|
| `DEVICE` | You probably should not change that                         |
| `VIDEO_CAPTURE_DEVICE` | Camera index (0 = webcam, 1 = external camera)              |
| `SAM2_CHECKPOINT` / `MODEL_CONFIG` | Path to the SAM2 checkpoint and its model config            |
| `FRAME_HEIGHT` / `FRAME_WIDTH` | Capture resolution, center-cropped; must be divisible by 32 |
| `YOLO_MODEL` | Base pretrained YOLO checkpoint to start from               |
| `LEARNING_RATE` | Optimizer learning rate for continual training              |
| `SAMPLE_BUFFER_SIZE` | Max samples kept per class for replay during training       |
| `SAMPLE_BATCH_SIZE` | Samples drawn per class per training step                   |
| `CONFIDENCE_EMA` / `EIGENCAM_EMA` | Smoothing factors to reduce frame-to-frame jitter           |
| `CONF_THRESHOLD` / `IOU_THRESHOLD` / `MAX_DETECTIONS` | Non-max suppression settings for prediction                 |

## Troubleshooting

- **`ImportError: cannot import name 'build_sam2_camera_predictor'`** usually means a *different* `sam2` package (e.g. the official non-real-time PyPI/GitHub version) already exists in your venv's `site-packages` and is shadowing the fork. Delete that folder (and its matching `.dist-info` folder if present) and re-verify with the check command in step 4.
- **No camera image (with external camera)**: try restarting the program