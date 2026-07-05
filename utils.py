import cv2
import numpy as np
import config as cfg

capture: cv2.VideoCapture | None = None


def get_camera_frame() -> tuple[bool, np.ndarray]:
    """
    Captures the current frame from the specified camera.
    Change the camera by using a different index for VIDEO_CAPTURE_DEVICE in config.py
    :return: boolean: active, frame
    """

    global capture

    # connect to camera if not yet connected
    if capture is None:
        capture = cv2.VideoCapture(cfg.VIDEO_CAPTURE_DEVICE)
        print("[INFO] Connected to camera")

    assert capture
    active, frame = capture.read()
    #cropping is necessary to ensure frame dimensions are divisible by 32 (required by YOLO)
    cropped_frame = crop_frame(frame)

    return active, cropped_frame


def crop_frame(frame):
    """
    Crops frame to the given resolution.
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

    # if no frame is given (camera booting up etc.)
    except AttributeError:
        return frame
