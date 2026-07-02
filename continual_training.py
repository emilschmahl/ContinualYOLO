import cv2
import torch
import tkinter as tk
from tkinter import simpledialog
import segment
import utils

# auto-tune cudnn to improve performance
torch.backends.cudnn.benchmark = True

tk_root = tk.Tk()
tk_root.withdraw()

selected_point = None
selected_class = None
class_selected = True
show_mask = True
show_box = True

def ask_class_name(prompt="Assign class to selected object:"):
    """
    Zeigt ein kleines Eingabefenster für den Klassennamen an.
    :param prompt: Text im Dialogfenster
    :return: eingegebener String, oder None, falls abgebrochen
    """
    tk_root.attributes("-topmost", True)  # Dialog vor das OpenCV-Fenster bringen
    result = simpledialog.askstring("Klasse", prompt, parent=tk_root)
    return result


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

    window_name = "Live Preview, press Q to quit, press M to toggle mask, press B to toggle box"
    cv2.startWindowThread()
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse_click)

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

                if not class_selected:
                    selected_class = ask_class_name()
                    class_selected = True

        finally:
            cv2.destroyAllWindows()