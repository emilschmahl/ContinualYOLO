import config as cfg
import cv2
import numpy as np
import torch


def get_camera_frame(camera: cv2.VideoCapture) -> tuple[bool, np.ndarray]:
    """
    Captures the current frame from the given camera.
    Change the camera by using a different index for VIDEO_CAPTURE_DEVICE in config.py
    :return: is_active, frame
    """

    is_active, frame = camera.read()

    #cropping is necessary to ensure frame dimensions are divisible by 32 (required by YOLO)
    cropped_frame = _crop_frame(frame)

    return is_active, cropped_frame


def _crop_frame(frame: np.ndarray) -> np.ndarray:
    """
    Crops frame to the given resolution (center crop).
    If the frame is smaller than the target resolution, a ValueError is raised.
    :param frame: camera frame
    :return: cropped frame
    """
    try:
        h, w = frame.shape[:2]

        if (cfg.FRAME_HEIGHT % 32 != 0) or (cfg.FRAME_WIDTH % 32 != 0):
            raise ValueError("[ERROR] CHANGE FRAME SIZE IN CONFIG Target height and width must be divisible by 32")

        elif h < cfg.FRAME_HEIGHT or w < cfg.FRAME_WIDTH:
            raise ValueError(
                f"[ERROR] CHANGE FRAME SIZE IN CONFIG Camera resolution {w}x{h} is smaller then the given target resolution {cfg.FRAME_WIDTH}x{cfg.FRAME_HEIGHT}"
            )

        x_start = (w - cfg.FRAME_WIDTH) // 2
        y_start = (h - cfg.FRAME_HEIGHT) // 2

        return frame[y_start:y_start + cfg.FRAME_HEIGHT, x_start:x_start + cfg.FRAME_WIDTH]

    # if no or empty frame is given
    except AttributeError:
        return frame


def build_batch(samples):
    """
    Creates input for v8DetectionLoss from list of samples:

    the batch must have the following format:

    batch = {
        ("img": images)\n
        "batch_idx": batch_idx,\n
        "cls": cls,\n
        "bboxes": bboxes,\n
    }
    :param samples: list of samples: Sample
    :return: batch
    """
    # convert numpy array to tensor (frames without class or bbox are used as negative samples)
    imgs = torch.stack([torch.from_numpy(s.frame).permute(2, 0, 1).float().div(255) for s in samples]).to(cfg.DEVICE)

    batch_idx, cls, bboxes = [], [], []
    for i, sample in enumerate(samples):

        # create no entry if no class is passed
        if sample.class_id is None:
            continue

        box = (sample.x_center, sample.y_center, sample.box_width, sample.box_height)
        # in some cases a class but no bbox was passed, use as negative sample
        if any(v is None for v in box):
            continue

        batch_idx.append(i)
        cls.append(sample.class_id)
        bboxes.append(list(box))

    return {
        "img": imgs,
        "batch_idx": torch.tensor(batch_idx, dtype=torch.float32, device=cfg.DEVICE),
        "cls": torch.tensor(cls, dtype=torch.float32, device=cfg.DEVICE).unsqueeze(1),
        "bboxes": torch.tensor(bboxes, dtype=torch.float32, device=cfg.DEVICE)
    }