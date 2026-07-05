import config as cfg
import numpy as np
import queue
import segment
import threading
import torch
from collections import defaultdict
from itertools import count
from numpy_ringbuffer import RingBuffer
from torch import nn
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


class YOLOTrainer:

    def __init__(self):

        # toggles training mode (may be renamed in later revisions)
        self.active = True

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

        self.thread = threading.Thread(target=self.wait_for_samples, daemon=True)
        # when called, asserts that only one thread can use the model (mutex)
        self.lock = threading.Lock()


    def start(self):
        """
        Starts the trainer in training mode.
        """
        self.active = True
        self.thread.start()


    def stop(self):
        """
        Stops the thread.
        """
        self.active = False
        self.thread.join(timeout=2)


    def wait_for_samples(self):
        """
        Awaits samples in sample_queue. Samples must be objects of Sample.
        """
        self.det_model.train()
        print("[INFO] PROGRAM RUNS IN TRAINING MODE")
        while self.active:
            try:
                sample: Sample = self.sample_queue.get()
            except queue.Empty:
                continue

            self.train_on_sample(sample)
            print(f"[QUEUE] {self.sample_queue.qsize()} samples in queue")


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
            mask = (mask > 0.0).squeeze().detach().cpu().numpy()
            bounding_box = segment.calculate_bounding_box(frame, mask)
            x_center, y_center, box_width, box_height = bounding_box

        except TypeError:
            # leave label file empty if no mask is passed
            class_id, x_center, y_center, box_width, box_height = [None] * 5

        sample = Sample(frame, class_name, x_center, y_center, box_width, box_height)

        self.sample_queue.put(sample)
        print(f"[QUEUE] {self.sample_queue.qsize()} samples in queue")


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