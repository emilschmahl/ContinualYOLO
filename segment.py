import numpy as np


def get_mask(predictor, frame: np.ndarray, selected_point: tuple[float, float] | None) -> np.ndarray:
    """
    Predict mask for frame and selected point.
    Set selected point to None.
    :param predictor: SAM2 model
    :param selected_point: tuple(x, y)
    :param frame: np.nparray containing the current frame
    :return: SAM2 mask
    """

    #start tracking the object at the selected point
    if selected_point is not None:
        predictor.load_first_frame(frame)
        _, _, mask = predictor.add_new_prompt(
            frame_idx=0,
            obj_id=1,
            points=np.array([selected_point], dtype=np.float32),
            labels=np.array([1]),
        )

    # track the object if no new point is passed
    else:
        _, mask = predictor.track(frame)

    """
    Transforms the raw SAM2 output into an usable np.ndarray.
    mask > 0.0: True if value > 0.0 else False
    squeeze:    removes dimensions of size 1 e.g. (1,1,H,W) to (H,W)
    detach:     detaches tensor from the autograd graph
    cpu:        moves the tensor from GPU to CPU memory
    numpy:      converts the tensor to a numpy array
    """
    mask = (mask > 0.0).squeeze().detach().cpu().numpy()
    return mask


def overlay_mask(frame: np.ndarray, mask: np.ndarray, color=(0, 255, 0), transparency=0.3):
    """
    Combines frame and mask to visualize the segmented area.
    :param frame: np.nparray containing the current frame
    :param mask: the matching SAM2 output
    :param color: RGB value of the mask (green by default)
    :param transparency: mask transparency from 0 to 1
    :return: a masked frame
    """

    color = np.array(color, dtype=np.uint8)
    m_frame = frame.copy()

    m_frame[mask] = (frame[mask] * (1 - transparency) + color * transparency).astype(np.uint8)
    return m_frame


def calculate_bounding_box(frame, mask):
    """
    Calculates bounding box parameters for a given frame and mask.
    The output can be used as parameters for a YOLO label.
    :param frame: np.nparray containing a frame
    :param mask: the matching SAM2 output
    :return: tuple(x_center, y_center, width, height)
    """

    height, width = frame.shape[:2]

    # collapse mask into 1D projections, true if any pixel in row/ col is part of the mask
    cols, rows = np.any(mask, axis=0), np.any(mask, axis=1)

    if not rows.any():
        return None

    # first and last row/ col containing the mask
    c0, c1 = np.where(cols)[0][[0, -1]]
    r0, r1 = np.where(rows)[0][[0, -1]]

    box_width, box_height = c1 - c0 + 1, r1 - r0 + 1
    x_center, y_center = c0 + (box_width / 2), r0 + (box_height / 2)

    # calculate values relative to frame size
    bounding_box = (x_center / width, y_center / height, box_width / width, box_height / height)
    return bounding_box


def overlay_box(frame, bounding_box, color=(0,255,0), thickness=2):
    """
    Combines frame and bounding box to visualize the segmented area.
    :param frame: np.nparray containing the current frame
    :param bounding_box: bounding box with values [0:1]
    :param color: RGB value of the bounding box (green by default)
    :param thickness: bounding box thickness
    :return: a copy of the frame with bounding box
    """

    overlay = frame.copy()
    if bounding_box is None:
        return overlay

    height, width = frame.shape[:2]
    x_center, y_center, box_width, box_height = bounding_box
    # calculate absolute values
    x_center, box_width = x_center * width, box_width * width
    y_center, box_height = y_center * height, box_height * height

    # calculate box corners
    x0, x1 = int(round(x_center - box_width / 2)), int(round(x_center + box_width / 2)) - 1
    y0, y1 = int(round(y_center - box_height / 2)), int(round(y_center + box_height / 2)) - 1

    # outer bounds of the border strips (clamped to frame edges)
    x0_b, x1_b = max(x0 - thickness, 0), min(x1 + 1 + thickness, width)
    y0_b, y1_b = max(y0 - thickness, 0), min(y1 + 1 + thickness, height)

    # top strip
    overlay[y0_b:y0, x0_b:x1_b] = color
    # bottom strip
    overlay[y1 + 1:y1_b, x0_b:x1_b] = color
    # left strip
    overlay[y0:y1 + 1, x0_b:x0] = color
    # right strip
    overlay[y0:y1 + 1, x1 + 1:x1_b] = color

    return overlay