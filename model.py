import os
import time
from collections import defaultdict
from itertools import count
import cv2
import segment
import config as cfg

class_indices = defaultdict(count().__next__)


def save_yolo_sample(frame, class_name, mask, dest_dir=cfg.DEST_DIR):
    """
    Saves a frame and its matching YOLO label to dict.
    Frame is saved under DEST_DIR/frame, label under DEST_DIR/label.
    :param frame: np.ndarray containing the current frame
    :param class_name: user-provided class name (str), mapped to a YOLO index internally
    :param mask: the matching SAM2 output for the frame
    :param dest_dir: base directory for the dataset
    """
    try:
        mask = (mask > 0.0).squeeze().detach().cpu().numpy()
        bounding_box = segment.calculate_bounding_box(frame, mask)
        x_center, y_center, box_width, box_height = bounding_box
        class_id = class_indices[class_name]

    except TypeError:
        # leave label file empty if no mask is passed
        class_id, x_center, y_center, box_width, box_height = [""] * 5

    frame_dir = os.path.join(dest_dir, "frame")
    label_dir = os.path.join(dest_dir, "label")
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    # nanosecond timestamp keeps filenames unique even at high frame rates
    sample_name = f"{class_name}_{time.time_ns()}"
    frame_path = os.path.join(frame_dir, f"{sample_name}.jpg")
    label_path = os.path.join(label_dir, f"{sample_name}.txt")

    cv2.imwrite(frame_path, frame)
    with open(label_path, "w") as label_file:
        label_file.write(f"{class_id} {x_center} {y_center} {box_width} {box_height}\n")