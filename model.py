import config as cfg
import numpy as np
import queue
import segment
import threading
import time
import torch
import utils
from collections import defaultdict
from itertools import count
from numpy_ringbuffer import RingBuffer
from torch import nn
from types import SimpleNamespace
from typing import cast
from ultralytics import YOLO
from ultralytics.nn import DetectionModel
from ultralytics.nn.modules import Detect
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.loss import v8DetectionLoss


class Sample:
    """
    Acts as input for the YOLO model.
    """
    def __init__(self, frame, class_name, x_center, y_center, box_width, box_height):
        self.frame = frame
        self.class_name = class_name
        self.class_id: int | None = None
        self.x_center = x_center
        self.y_center = y_center
        self.box_width = box_width
        self.box_height = box_height


class LatestValue:
    """
    Saves the most recent set values. Values are deleted after accessing.
    """
    def __init__(self):
        self.values: SimpleNamespace | None = None
        self.lock = threading.Lock()

    def set(self, **kwargs):
        with self.lock:
            self.values = SimpleNamespace(**kwargs)

    def get(self):
        with self.lock:
            values = self.values
            self.values = None
            return values


class YOLOTrainer:
    """
    Manages continual learning of a YOLO model in a background thread that
    switches between training and prediction mode.

    Training mode:
        Consumes samples from sample_queue, buffers them in sample_buffer, and
        trains on a randomly assembled batch for each sample. New classes
        automatically expand the last detection layer; all other layers
        stay frozen.

    Prediction mode:
        Continuously reads the latest frame and writes the prediction to
        self.predicted_frame.
    """

    def __init__(self):

        # stops thread if False
        self.active = True
        # toggles training mode
        self.training = True

        # EMA-smoothed confidence to reduce frame-to-frame jitter from
        # per-frame argmax anchor selection (no NMS/tracking in place)
        self._confidence_ema: float | None = None

        # latest frame for prediction input (no queue is used to avoid an overflow due to lagging)
        self.frame = LatestValue()
        self.predicted_frame = LatestValue()

        # temporarily saves samples to create training batches (save samples of all trained classes)
        self.per_class_buffers: dict[int, RingBuffer] = defaultdict(
            lambda: RingBuffer(capacity=cfg.SAMPLE_BUFFER_SIZE, dtype=object)
        )
        # holds training sample input from the main thread
        self.sample_queue = queue.Queue()

        self.model = YOLO(cfg.YOLO_MODEL)
        self.model.to(cfg.DEVICE)

        try:
            checkpoint = torch.load(cfg.CONTINUAL_MODEL, map_location=cfg.DEVICE)
            self._load(checkpoint)
            print(f"[INFO] EXISTING MODEL WAS LOADED ({checkpoint["nc"]} classes)")

        except FileNotFoundError:
            self._load(self._empty_checkpoint())
            print("[INFO] NO MODEL FOUND Created empty model.")

        self.thread = threading.Thread(target=self._run, daemon=True)
        # when called, asserts that only one thread can use the model (mutex)
        self.lock = threading.Lock()


    def _empty_checkpoint(self) -> dict:
        """
        Represents a fresh start with no prior training history.
        Builds a model with nc=1 and transfers the pretrained YOLO weights via
        DetectionModel.load()
        """
        fresh_model = DetectionModel(cfg=self.model.model.yaml, nc=1, verbose=False).to(cfg.DEVICE)
        fresh_model.load(cast(nn.Module, self.model.model))

        return {
            "model_state_dict": fresh_model.state_dict(),
            "yaml": self.model.model.yaml,
            "nc": 1,
            "classes": {},
        }


    def _sync_names(self):
        """Save names from dict."""
        self.det_model.names = {class_id: name for name, class_id in self.classes.items()}


    def _rebuild_optimizer_and_criterion(self):
        """Rebuild optimizer and criterion on new parameters."""
        trainable = filter(lambda p: p.requires_grad, self.det_model.parameters())
        self.optimizer = torch.optim.AdamW(trainable, lr=cfg.LEARNING_RATE)
        # V8DetectionLoss is required by YOLO
        self.criterion = v8DetectionLoss(self.det_model)


    def _load(self, checkpoint):
        """
        Restores model weights and self.classes from a checkpoint.
        :param checkpoint: dict as produced by save() or _empty_checkpoint()
        """

        print("LOADING...")
        # build model for number of classes and load saved weights
        nc = checkpoint["nc"]
        self.det_model = DetectionModel(cfg=checkpoint["yaml"], nc=nc, verbose=False).to(cfg.DEVICE)
        self.det_model.load_state_dict(checkpoint["model_state_dict"])
        self.model.model = self.det_model

        # set default arguments
        merged = {**DEFAULT_CFG_DICT}
        self.det_model.args = IterableSimpleNamespace(**merged)

        # set class names and add already saved classes to dict
        saved_classes = checkpoint["classes"]
        self.classes = defaultdict(count(len(saved_classes)).__next__)
        self.classes.update(saved_classes)
        self._sync_names()

        # load class buffer for previously learned classes
        self.per_class_buffers = defaultdict(
            lambda: RingBuffer(capacity=cfg.SAMPLE_BUFFER_SIZE, dtype=object)
        )
        for class_id, samples in checkpoint.get("per_class_buffers", {}).items():
            buf = self.per_class_buffers[class_id]
            for sample in samples:
                buf.append(sample)

        # freeze all but the last layer
        self.det_model.requires_grad_(False)
        detect = cast(Detect, cast(object, self.det_model.model[-1]))
        detect.cv3.requires_grad_(True)

        self._rebuild_optimizer_and_criterion()


    def _save(self):
        """
        Saves the current model state to a .pt file: weights, class count,
        and the name->id mapping needed to restore self.classes on load.
        """

        print("SAVING...")

        detect = cast(Detect, cast(object, self.det_model.model[-1]))

        buffers_to_save = {
            class_id: list(buf) for class_id, buf in self.per_class_buffers.items()
        }

        torch.save({
            "model_state_dict": self.det_model.state_dict(),
            "yaml": self.det_model.yaml,
            "nc": detect.nc,
            "classes": dict(self.classes),
            "per_class_buffers": buffers_to_save,
        }, cfg.CONTINUAL_MODEL)

        print(f"[INFO] MODEL WAS SAVED to {cfg.CONTINUAL_MODEL}")


    def _run(self):
        """Toggles train and predict modes."""

        while self.active:

            if self.training:

                self.det_model.train()
                print("[INFO] PROGRAM RUNS IN TRAINING MODE")
                while self.training and self.active:

                    try:
                        sample: Sample = self.sample_queue.get(timeout=0.5)
                    except queue.Empty:
                        continue

                    self._train_on_sample(sample)
                    print(f"[QUEUE] {self.sample_queue.qsize()} samples in queue")

            elif not self.training:

                self.det_model.eval()
                print("[INFO] PROGRAM RUNS IN PREDICTION MODE")
                while not self.training and self.active:

                    self._predict_one()
                    time.sleep(0.01)

        self._save()


    def _train_on_sample(self, sample: Sample):
        """
        Trains the model with passed sample.
        Get the class id for the class name, add the sample to the buffer and randomly choose a batch.
        Then use the batch to adjust the model weights.
        :param sample: a training sample
        """
        sample.class_id = self._get_class_id(sample.class_name)

        # store sample in buffer
        if sample.class_id is not None:
            self.per_class_buffers[sample.class_id].append(sample)

        batch_samples = [sample]
        for class_id, buf in self.per_class_buffers.items():
            if len(buf) == 0:
                continue
            n = min(cfg.SAMPLE_BATCH_SIZE, len(buf))
            # get SAMPLE_BATCH_SIZE batches from all classes if possible
            index = np.random.choice(len(buf), size=n, replace=False)
            batch_samples += [buf[i] for i in index]

        batch = utils.build_batch(batch_samples)

        with self.lock:

            # forward-pass
            prediction = self.det_model(batch["img"])
            # calculate loss
            loss, _ = self.criterion(prediction, batch)
            # delete old gradients
            self.optimizer.zero_grad()
            # calculate new gradients
            loss.sum().backward()
            # adjust weights
            self.optimizer.step()


    def _get_class_id(self, class_name):
        """
        Returns the class id for the given class name.
        Creates a new class id if the class name is called for the first time.
        :param class_name: Sample.class_name
        :returns: integer class_id or None
        """
        # no bounding_box in frame
        if class_name is None:
            return None

        is_new = class_name not in self.classes

        # get id or create id if not exists
        class_id = self.classes[class_name]

        # append class to model if class is new and not the first class
        if is_new and len(self.classes) > 1:
            self._add_class()

        return class_id


    def _add_class(self):
        """
        Expand the last layer of the model, if a new class is given.
        Already trained classes are copied, the new class is initialized randomly and with a strong negative bias.
        """

        head = cast(Detect, cast(object, self.det_model.model[-1]))
        classes = head.nc
        new_classes = classes + 1

        # YOLO11 has one cv3 branch per detection scale (small/medium/large objects),
        # each ending in a 1x1 conv whose output channels equal the class count.
        # Adding a class means rebuilding that final conv at the new, larger size.
        for cv3_seq in head.cv3:
            old_conv = cv3_seq[-1]
            new_conv = torch.nn.Conv2d(old_conv.in_channels, new_classes, kernel_size=1).to(cfg.DEVICE)
            assert new_conv.bias is not None
            with torch.no_grad():
                # copy the learned weights for existing classes
                new_conv.weight[:classes], new_conv.bias[:classes] = old_conv.weight, old_conv.bias
                # initialize new class with small random weight
                torch.nn.init.normal_(new_conv.weight[classes:], std=0.01)
                # and a strong negative bias
                torch.nn.init.constant_(new_conv.bias[classes:], -4.6)
            cv3_seq[-1] = new_conv

        # sync new class count
        head.nc = self.det_model.nc = new_classes
        yaml_dict = cast(dict, getattr(self.det_model, "yaml", None))
        if yaml_dict:
            yaml_dict["nc"] = new_classes

        self._sync_names()
        self._rebuild_optimizer_and_criterion()

        print(f"[INFO] Added new class to model: {classes} -> {new_classes} classes")


    def _predict_one(self):
        """
        Predicts the safest prediction in the frame. (Cannot predict more than one object)
        Uses self.frame as input and self.predicted_frame as output.
        """
        frame = self.frame.get()
        if frame is None:
            return

        frame = frame.frame
        # convert numpy array to tensor (frames without class or bbox are used as negative samples)
        img = torch.from_numpy(frame).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(cfg.DEVICE)

        with self.lock, torch.no_grad():
            raw_predictions = self.det_model(img)[0]

        predictions = raw_predictions.squeeze(0)
        boxes, class_scores = predictions[:4], predictions[4:]

        # pick the single anchor point with the highest confidence across all classes
        best_idx = class_scores.amax(dim=0).argmax()
        best_class = int(class_scores[:, best_idx].argmax())

        cx, cy, w, h = boxes[:, best_idx].tolist()
        # probabilities for all classes
        scores = class_scores[:, best_idx].tolist()

        #softmax_scores = F.softmax(class_scores[:, best_idx], dim=0).tolist()

        raw_confidence = scores[best_class]
        best_class_name = self.det_model.names.get(best_class, str(best_class))

        if self._confidence_ema is None:
            self._confidence_ema = raw_confidence
        else:
            a = cfg.CONFIDENCE_EMA
            self._confidence_ema = a * self._confidence_ema + (1 - a) * raw_confidence

        self.predicted_frame.set(
            frame=frame,
            x_center=cx / cfg.FRAME_WIDTH,
            y_center=cy / cfg.FRAME_HEIGHT,
            box_width=w / cfg.FRAME_WIDTH,
            box_height=h / cfg.FRAME_HEIGHT,
            class_scores=scores,
            class_name=best_class_name,
            confidence=self._confidence_ema,
        )


    def start(self):
        """Starts the trainer in training mode."""
        self.thread.start()


    def stop(self):
        """Stops the thread."""
        self.active = False
        self.thread.join()
        return True


    def train_mode(self):
        """Sets mode to train"""
        self.training = True


    def predict_mode(self):
        """Sets mode to predict"""
        self.training = False


    def save_yolo_sample(self, frame, class_name, mask):
        """
        Saves a new sample in the input queue.
        :param frame: np.ndarray containing the current frame
        :param class_name: user-provided class name (str), mapped to a YOLO index internally
        :param mask: the matching SAM2 output for the frame
        """
        try:
            bounding_box = segment.calculate_bounding_box(frame, mask)
            x_center, y_center, box_width, box_height = bounding_box

        except TypeError:
            # leave label file empty if no mask is passed
            x_center, y_center, box_width, box_height = [None] * 4

        sample = Sample(frame, class_name, x_center, y_center, box_width, box_height)

        self.sample_queue.put(sample)
        print(f"[QUEUE] {self.sample_queue.qsize()} samples in queue")


    def set_frame(self, frame):
        self.frame.set(frame=frame)