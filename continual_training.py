import cv2
import torch
import utils
import numpy as np
import config as cfg
from sam2.build_sam import build_sam2_camera_predictor

# auto-tune cudnn to improve performance
torch.backends.cudnn.benchmark = True
# create a pre-trained instance of SAM2
predictor = build_sam2_camera_predictor(cfg.MODEL_CONFIG, cfg.SAM2_CHECKPOINT)

mask = None
selected_point = None
if_init = False


def set_selected_point(event, x, y, *_):
    """
    saves the mouse coordinates if left button is clicked
    :param event: mouse event
    :param x: x coordinate
    :param y: y coordinate
    """

    global selected_point

    if event == cv2.EVENT_LBUTTONDOWN:
        selected_point = (x, y)


def overlay_mask(frame, mask, color=(0, 255, 0), transparency=0.3):
    """
    shade the selected object in a different color
    :param frame: the current frame
    :param mask: the SAM2 mask
    :param color: the shading color
    :param transparency: transparency of color
    :return: the masked frame
    """

    mask = (mask > 0.0).squeeze().detach().cpu().numpy()

    color = np.array(color, dtype=np.uint8)
    overlay = frame.copy()
    overlay[mask] = (overlay[mask] * (1 - transparency) + color * transparency).astype(np.uint8)
    return overlay


if __name__ == "__main__":

    window_name = "Live Preview, press q to quit"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, set_selected_point)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        try:
            while True:
                active, frame = utils.get_camera_frame()
                if not active:
                    print(f"\033[91m[ERROR] NO FRAME WAS PASSED. Is the camera connected and running?\033[0m"f"")
                    break

                # mask the object if user has selected an object
                if selected_point is not None:
                    predictor.load_first_frame(frame)
                    _, _, mask = predictor.add_new_prompt(
                        frame_idx=0,
                        obj_id=1,
                        points=np.array([selected_point], dtype=np.float32),
                        labels=np.array([1]),
                    )
                    if_init = True
                    selected_point = None

                elif if_init:
                    _, mask = predictor.track(frame)

                preview = frame if mask is None else overlay_mask(frame, mask)
                cv2.imshow(window_name, preview)

                if cv2.waitKey(1) == ord('q'):
                    break

        finally:
            cv2.destroyAllWindows()