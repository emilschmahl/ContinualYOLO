import config as cfg
import numpy as np
import queue
import segment
import threading
import time
import torch
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

    def __init__(self):

        # stops thread if False
        self.active = True
        # toggles training mode
        self.training = True

        # latest frame for prediction input (no queue is used to avoid an overflow due to lagging)
        self.frame = LatestValue()
        self.predicted_frame = LatestValue()

        self.classes = defaultdict(count().__next__)
        # temporarily saves samples to create training batches
        self.sample_buffer = RingBuffer(capacity=cfg.SAMPLE_BUFFER_SIZE, dtype=object)
        # holds sample input from the main thread
        self.sample_queue = queue.Queue()

        self.model = YOLO(cfg.YOLO_MODEL)
        self.model.to(cfg.DEVICE)


        existing_args = getattr(self.model.model, "args", {})
        if not isinstance(existing_args, dict):
            existing_args = {}

        # reset classes (last detection layer)
        self.det_model = DetectionModel(cfg=self.model.model.yaml, nc=1, verbose=False).to(cfg.DEVICE)
        self.det_model.load(cast(nn.Module, self.model.model))
        self.det_model.names = {}
        self.model.model = self.det_model

        merged = {**DEFAULT_CFG_DICT, **existing_args}
        self.det_model.args = IterableSimpleNamespace(**merged)

        # freeze all layers
        self.det_model.requires_grad_(False)

        # unfreeze last layer
        detect_head = cast(object, self.det_model.model[-1])
        detect_head.cv3.requires_grad_(True)

        # only optimize weights that are not frozen
        trainable = filter(lambda p: p.requires_grad, self.det_model.parameters())
        self.optimizer = torch.optim.AdamW(trainable, lr=cfg.LEARNING_RATE)

        self.criterion = v8DetectionLoss(self.det_model)

        self.thread = threading.Thread(target=self.run, daemon=True)
        # when called, asserts that only one thread can use the model (mutex)
        self.lock = threading.Lock()


    def start(self):
        """
        Starts the trainer in training mode.
        """
        self.thread.start()


    def stop(self):
        """
        Stops the thread.
        """
        self.active = False
        self.thread.join()
        return True


    def train_mode(self):
        """
        Sets mode to train
        """
        self.training = True


    def predict_mode(self):
        """
        Sets mode to predict
        """
        self.training = False


    def run(self):
        """
        Toggles train and predict modes.
        When in train mode:
        Awaits samples in sample_queue. Samples must be objects of Sample.
        When in predict mode:
        Tries to predict objects in frame.
        """

        while self.active:
            if self.training:
                self.load()
                self.det_model.train()
                print("[INFO] PROGRAM RUNS IN TRAINING MODE")
                while self.training and self.active:

                    try:
                        sample: Sample = self.sample_queue.get(timeout=0.5)
                    except queue.Empty:
                        continue

                    self.train_on_sample(sample)
                    print(f"[QUEUE] {self.sample_queue.qsize()} samples in queue")

            elif not self.training:
                self.save()
                self.det_model.eval()
                print("[INFO] PROGRAM RUNS IN PREDICTION MODE")
                while not self.training and self.active:
                    self.predict_one()
                    time.sleep(0.01)

        self.save()


    def predict_one(self):
        """
        Predicts the safest prediction in the frame. (Cannot predict more than one object)
        Uses self.frame as input and self.predicted_frame as output.
        """
        frame = self.frame.get()
        if frame is None:
            return

        frame = frame.frame
        # convert numpy array to tensor
        img = torch.from_numpy(frame).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(cfg.DEVICE)

        with self.lock, torch.no_grad():
            raw_predictions = self.det_model(img)[0]

        predictions = raw_predictions.squeeze(0)
        boxes, class_scores = predictions[:4], predictions[4:]

        # pick the single anchor point with the highest confidence across all classes
        best_idx = class_scores.amax(dim=0).argmax()

        cx, cy, w, h = boxes[:, best_idx].tolist()
        # probabilities for all classes
        scores = class_scores[:, best_idx].tolist()

        self.predicted_frame.set(
            frame=frame,
            x_center=cx / cfg.FRAME_WIDTH,
            y_center=cy / cfg.FRAME_HEIGHT,
            box_width=w / cfg.FRAME_WIDTH,
            box_height=h / cfg.FRAME_HEIGHT,
            class_scores=scores,
        )


    def save(self):
        """
        Saves the current model state to a .pt file: weights, class count,
        and the name->id mapping needed to restore self.classes on load.
        """
        print("[INFO] SAVING MODEL...")

        detect = cast(Detect, cast(object, self.det_model.model[-1]))
        torch.save({
            "model_state_dict": self.det_model.state_dict(),
            "yaml": self.det_model.yaml,
            "nc": detect.nc,
            "classes": dict(self.classes),
        }, cfg.CONTINUAL_MODEL)

        print(f"[INFO] MODEL WAS SAVED to {cfg.CONTINUAL_MODEL}")


    def load(self):
        """
        Restores model weights and self.classes from a checkpoint saved by save().
        Rebuilds the class name->id counter so it continues from the correct
        next id, rather than restarting at 0.
        """
        try:
            checkpoint = torch.load(cfg.CONTINUAL_MODEL, map_location=cfg.DEVICE)

        except FileNotFoundError:
            print("[INFO] NO MODEL FOUND")
            return

        # build model for number of classes
        nc = checkpoint["nc"]
        self.det_model = DetectionModel(cfg=checkpoint["yaml"], nc=nc, verbose=False).to(cfg.DEVICE)
        self.det_model.load_state_dict(checkpoint["model_state_dict"])
        self.model.model = self.det_model

        # set default arguments
        merged = {**DEFAULT_CFG_DICT}
        self.det_model.args = IterableSimpleNamespace(**merged)

        # set class names
        saved_classes = checkpoint["classes"]
        self.classes = defaultdict(count(len(saved_classes)).__next__)
        self.classes.update(saved_classes)
        self.det_model.names = {class_id: name for name, class_id in self.classes.items()}

        # freeze all but the last layer
        self.det_model.requires_grad_(False)
        detect = cast(Detect, cast(object, self.det_model.model[-1]))
        detect.cv3.requires_grad_(True)

        # initialize optimizer and criterion
        trainable = filter(lambda p: p.requires_grad, self.det_model.parameters())
        self.optimizer = torch.optim.AdamW(trainable, lr=cfg.LEARNING_RATE)
        self.criterion = v8DetectionLoss(self.det_model)

        print(f"[INFO] EXISTING MODEL WAS LOADED ({nc} classes)")


    def train_on_sample(self, sample: Sample):
        """
        Trains the model with passed sample.
        Get the class id for the class name, add the sample to the buffer and randomly choose a batch.
        Then use the batch to adjust the model weights.
        :param sample: a training sample
        """
        sample.class_id = self.get_class_id(sample.class_name)

        self.sample_buffer.append(sample)

        batch_samples = [sample]
        if len(self.sample_buffer) > 1:
            n = min(cfg.SAMPLE_BATCH_SIZE, len(self.sample_buffer) - 1)
            # randomly pick samples from the buffer to be used in the batch
            index = np.asarray(np.random.choice(len(self.sample_buffer) - 1, size=n, replace=False))
            batch_samples += [self.sample_buffer[i] for i in index]

        batch = build_batch(batch_samples)

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


    def get_class_id(self, class_name):
        """
        Returns the class id for the given class name.
        Creates a new class id if the class name is called for the first time.
        :param class_name: Sample.class_name
        :returns: integer class_id
        """
        # no bounding_box in frame
        if class_name is None:
            return None

        is_new = class_name not in self.classes

        # get id or create id if not exists
        class_id = self.classes[class_name]

        # append class to model if class is new and not the first class
        if is_new and len(self.classes) > 1:
            self.add_class()

        return class_id


    def add_class(self):
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

        # set the name for each class
        self.det_model.names = {class_id: name for name, class_id in self.classes.items()}

        # rebuild optimizer and criterion
        trainable = filter(lambda p: p.requires_grad, self.det_model.parameters())
        self.optimizer = torch.optim.AdamW(trainable, lr=self.optimizer.param_groups[0]["lr"])
        self.criterion = v8DetectionLoss(self.det_model)

        print(f"[INFO] Added new class to model: {classes} -> {new_classes} classes")


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
            class_id, x_center, y_center, box_width, box_height = [None] * 5

        sample = Sample(frame, class_name, x_center, y_center, box_width, box_height)

        self.sample_queue.put(sample)
        print(f"[QUEUE] {self.sample_queue.qsize()} samples in queue")


    def set_frame(self, frame):
        self.frame.set(frame=frame)


def build_batch(samples):
    """
    Input for v8DetectionLoss:

    Example for 2 samples in batch:
        sample 0: frame A, class 3, bbox
        sample 1: frame B, class 5, bbox
        -> img       = [A, B]
        -> batch_idx = [0, 1]
        -> cls       = [3, 5]
        -> bboxes    = [[...], [...]]
    """
    # convert numpy array to tensor
    imgs = torch.stack([torch.from_numpy(s.frame).permute(2, 0, 1).float() / 255.0 for s in samples]).to(cfg.DEVICE)

    batch_idx, cls, bboxes = [], [], []
    for i, sample in enumerate(samples):

        # create no entry if no class is passed (no mask in frame)
        if sample.class_id is None:
            continue
        batch_idx.append(i)
        cls.append(sample.class_id)
        bboxes.append([sample.x_center, sample.y_center, sample.box_width, sample.box_height])

    return {
        "img": imgs,
        "batch_idx": torch.tensor(batch_idx, dtype=torch.float32, device=cfg.DEVICE),
        "cls": torch.tensor(cls, dtype=torch.float32, device=cfg.DEVICE).view(-1, 1),
        "bboxes": torch.tensor(bboxes, dtype=torch.float32, device=cfg.DEVICE)
    }