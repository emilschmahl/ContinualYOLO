import config as cfg
import cv2
import queue
import segment
import threading
import time
import torch
import utils
from model import YOLOTrainer
from sam2.build_sam import build_sam2_camera_predictor
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="sam2.sam2_camera_predictor"
)

class_request_queue = queue.Queue()
class_result_queue = queue.Queue()

has_started_sam2 = False
selected_point = None
selected_class = None
class_selected = True
waiting_for_class = False
show_mask = True
show_box = True
recording = False
mask = None


def console_input_worker():
    """
    Waits for class request, then takes class name from user.
    This function runs in its own thread to not interrupt the main function.
    """

    while True:
        prompt = class_request_queue.get()
        if prompt is None:
            continue
        result = input(f"{prompt} ").strip()
        class_result_queue.put(result if result else None)


def on_mouse_click(event, x, y, *_):
    """
    Saves the mouse position on left click and takes the class name from the user.
    Class name is used to label YOLO training data.
    :param event: mouse event
    :param x: mouse x coordinate
    :param y: mouse y coordinate
    """

    global class_selected, selected_point, has_started_sam2

    if event == cv2.EVENT_LBUTTONDOWN:
        selected_point = (x, y)
        has_started_sam2 = True
        class_selected = False


if __name__ == "__main__":

    window_name = "Continual YOLO"
    cv2.startWindowThread()
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse_click)

    # create a pre-trained instance of SAM2
    predictor = build_sam2_camera_predictor(cfg.MODEL_CONFIG, cfg.SAM2_CHECKPOINT, device=cfg.DEVICE)

    capture = cv2.VideoCapture(cfg.VIDEO_CAPTURE_DEVICE)

    print("[INFO] Connected to camera")

    threading.Thread(target=console_input_worker, daemon=True).start()

    trainer = YOLOTrainer()
    trainer.start()

    with torch.inference_mode(), torch.autocast(cfg.DEVICE, dtype=torch.float16):
        try:
            while True:
                active, frame = utils.get_camera_frame(capture)
                if not active:
                    print(f"\033[91m[ERROR] NO FRAME WAS PASSED. Is the camera connected and running?\033[0m"f"")
                    break

                if trainer.training:
                    if has_started_sam2:
                        mask = segment.get_mask(predictor, frame, selected_point)
                        selected_point = None

                        frame = segment.overlay_mask(frame, mask) if (mask is not None and show_mask) else frame
                        frame = segment.overlay_box(frame, segment.calculate_bounding_box(frame, mask)) if (mask is not None and show_box) else frame

                    class_label = f"Class: {selected_class}" if selected_class is not None else "Class: (none)"
                    cv2.putText(
                        frame,
                        class_label,
                        (10, 30), #position
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, #text size
                        (0, 255, 0), # color
                        2, #thickness
                        cv2.LINE_AA
                        )

                    cv2.imshow(window_name, frame)

                    if recording and  mask is not None:
                        trainer.save_yolo_sample(frame, selected_class, mask)
                        time.sleep(0.1)

                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        trainer.stop()
                        break
                    elif key == ord('m'):
                        show_mask = not show_mask
                    elif key == ord('b'):
                        show_box = not show_box
                    elif key == ord("s"):
                        if mask is not None:
                            trainer.save_yolo_sample(frame, selected_class, mask)
                    elif key == ord("e"):
                        trainer.training = not trainer.training
                    elif key == ord("r") and mask is not None:
                        recording = not recording
                        if recording:
                            print("RECORDING STARTED")
                        else:
                            print("RECORDING STOPPED")

                    if not class_selected and not waiting_for_class:
                        class_request_queue.put("Assign class to selected object:")
                        waiting_for_class = True

                    if waiting_for_class:
                        try:
                            selected_class = class_result_queue.get_nowait()
                            class_selected = True
                            waiting_for_class = False
                        except queue.Empty:
                            pass

                else:
                    trainer.set_frame(frame)
                    prediction = trainer.predicted_frame.get()
                    if prediction is not None:
                        masked_frame = prediction.frame

                        height, width = masked_frame.shape[:2]

                        for i, detection in enumerate(prediction.detections):
                            masked_frame = segment.overlay_box(masked_frame, (
                                detection.x_center,
                                detection.y_center,
                                detection.box_width,
                                detection.box_height
                            ))

                            box_top = (detection.y_center - detection.box_height / 2) * height
                            box_left = (detection.x_center - detection.box_width / 2) * width

                            label = f"{detection.class_name}: {detection.confidence * 100:.1f}%"
                            (text_width, text_height), _ = cv2.getTextSize(
                                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                            )

                            text_x = int(max(0, min(box_left, width - text_width)))
                            text_y = int(max(text_height + 4, box_top - 6))

                            cv2.putText(
                                masked_frame,
                                label,
                                (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 0),
                                2,
                                cv2.LINE_AA
                            )

                        cv2.imshow(window_name, masked_frame)
                    else:
                        cv2.imshow(window_name, frame)

                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        trainer.stop()
                        break
                    elif key == ord("e"):
                        trainer.training = not trainer.training


        finally:
            cv2.destroyAllWindows()