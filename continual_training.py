import cv2
import torch
import tkinter as tk
from tkinter import simpledialog
import threading
import queue
import segment
import utils
import model

# auto-tune cudnn to improve performance
torch.backends.cudnn.benchmark = True

class_request_queue = queue.Queue()
class_result_queue = queue.Queue()

selected_point = None
selected_class = None
class_selected = True
waiting_for_class = False
show_mask = True
show_box = True


def tkinter_worker():
    """
    Waits for class request, then takes class name from user.
    This function runs in its own thread to not interrupt the main function.
    """
    root = tk.Tk()
    root.withdraw()

    while True:
        prompt = class_request_queue.get()
        if prompt is None:
            continue
        root.attributes("-topmost", True)
        result = simpledialog.askstring("", prompt, parent=root)
        class_result_queue.put(result)


def on_mouse_click(event, x, y, *_):
    """
    Saves the mouse position on left click and takes the class name from the user.
    Class name is used to label YOLO training data.
    :param event: mouse event
    :param x: mouse x coordinate
    :param y: mouse y coordinate
    """

    global class_selected, selected_point

    if event == cv2.EVENT_LBUTTONDOWN:
        selected_point = (x, y)
        class_selected = False


if __name__ == "__main__":

    window_name = "Live Preview, press Q to quit, press M to toggle mask, press B to toggle box, press S to save sample"
    cv2.startWindowThread()
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse_click)

    threading.Thread(target=tkinter_worker, daemon=True).start()

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        try:
            while True:
                active, frame = utils.get_camera_frame()
                if not active:
                    print(f"\033[91m[ERROR] NO FRAME WAS PASSED. Is the camera connected and running?\033[0m"f"")
                    break

                mask, selected_point = segment.get_mask(frame, selected_point)

                masked_frame = segment.overlay_mask(frame, mask) if (mask is not None and show_mask) else frame
                boxed_frame = segment.overlay_box(masked_frame, mask) if (mask is not None and show_box) else masked_frame

                cv2.imshow(window_name, boxed_frame)

                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                elif key == ord('m'):
                    show_mask = not show_mask
                elif key == ord('b'):
                    show_box = not show_box
                elif key == ord("s"):
                    model.save_yolo_sample(frame, selected_class, mask)

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

        finally:
            cv2.destroyAllWindows()