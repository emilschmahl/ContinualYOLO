import cv2
import numpy as np
import config as cfg

_capture: cv2.VideoCapture | None = None

def get_camera_frame() -> tuple[bool, np.ndarray]:
    """
    Captures the current frame from the specified camera.
    Change the camera by using a different index for VIDEO_CAPTURE_DEVICE in config.py
    :return: boolean: active, frame
    """

    global _capture

    # connect to camera if not yet connected
    if _capture is None:
        _capture = cv2.VideoCapture(cfg.VIDEO_CAPTURE_DEVICE)

    return _capture.read()


def show_live_preview():
    """
    Opens a window and displays the camera stream
    """

    try:
        while True:

            active, frame = get_camera_frame()

            if not active:
                print(f"\033[91m[ERROR] NO FRAME WAS PASSED. Is the camera connected and running?\033[0m"f"")
                break

            cv2.imshow("Live Preview, press q to quit", frame)
            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()

