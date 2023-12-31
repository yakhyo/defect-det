import os

import random

import numpy as np
import torch
from torchvision.transforms import functional as F
from torchvision import transforms as T

from enum import Enum

from utils import LOGGER


class LossReduction:
    """Alias for loss reduction"""

    NONE = "none"
    MEAN = "mean"
    SUM = "sum"


def weight_reduce_loss(loss, weight=None, reduction: LossReduction = "mean"):
    """Apply element-wise weight and reduce loss.
    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if reduction == LossReduction.MEAN:
        loss = torch.mean(loss)
    elif reduction == LossReduction.SUM:
        loss = torch.sum(loss)
    elif reduction == LossReduction.NONE:
        return loss

    return loss


class EvalTransform:
    """Evaluation Augmentation"""

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) -> None:
        self.transforms = Compose(
            [
                PILToTensor(),
                ConvertImageDtype(torch.float),
                Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img, target):
        return self.transforms(img, target)


class TrainTransform:

    def __init__(
            self,
            use_random_perspective: bool = False,
            hflip_prop: float = 0.5,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
    ) -> None:
        transforms = []
        if hflip_prop > 0:
            transforms.append(RandomHorizontalFlip(hflip_prop))
        if use_random_perspective:
            transforms.append(RandomPerspective())
        transforms.extend(
            [
                PILToTensor(),
                ConvertImageDtype(torch.float),
                Normalize(mean=mean, std=std)
            ]
        )
        self.transforms = Compose(transforms)

    def __call__(self, img, target):
        return self.transforms(img, target)


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class RandomPerspective:
    def __init__(self):
        self.func = T.RandomPerspective()

    def __call__(self, image, target):
        image = self.func(image)
        target = self.func(target)
        return image, target


class PILToTensor:
    """Convert PIL image to torch tensor"""

    def __call__(self, image, target):
        image = F.pil_to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class ConvertImageDtype:
    """Convert Image dtype"""

    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, image, target):
        image = F.convert_image_dtype(image, self.dtype)
        return image, target


class Compose:
    """Composing all transforms"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip:
    """Random horizontal flip"""

    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


def add_weight_decay(model, weight_decay=1e-5):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': weight_decay}]


def strip_optimizers(f: str, s: str):
    """Strip optimizer from 'f' to finalize training"""
    x = torch.load(f, map_location="cpu")
    for k in "optimizer", "best_score":
        x[k] = None
    x["epoch"] = -1
    x["model"].half()  # to FP16
    for p in x["model"].parameters():
        p.requires_grad = False
    torch.save(x, s)
    mb = os.path.getsize(s) / 1e6  # get file size
    LOGGER.info(f"Optimizer stripped from {f}, saved as {s} {mb:.1f}MB")


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    # Simple early stopper
    def __init__(self, patience=10):
        self.best_fitness = 0.0  # i.e. mIOU
        self.best_epoch = 0
        self.patience = patience or float('inf')  # epochs to wait after fitness stops improving to stop

    def __call__(self, epoch, fitness):
        if fitness >= self.best_fitness:
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch  # epochs without improvement
        stop = delta >= self.patience  # stop training if patience exceeded
        if stop:
            LOGGER.info(f'Stopping training early as no improvement observed in last {self.patience} epochs. '
                        f'Best results observed at epoch {self.best_epoch}, best model saved as best.pt')
        return stop


def jaccard_index(inputs, target, num_classes):
    # Convert the prediction and target tensors to one-hot encoding
    inputs = torch.softmax(inputs, dim=1)
    inputs = torch.argmax(inputs, dim=1)
    inputs = torch.nn.functional.one_hot(inputs, num_classes=num_classes)
    target = torch.nn.functional.one_hot(target, num_classes=num_classes)

    # Calculate the intersection and union tensors
    intersection = (inputs & target).float().sum((0, 1, 2))
    union = (inputs | target).float().sum((0, 1, 2))

    # Calculate the Jaccard Index for each class
    jaccard = intersection / (union + 1e-15)

    return jaccard
