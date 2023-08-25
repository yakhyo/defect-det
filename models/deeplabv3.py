import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

model = deeplabv3_resnet50(models=True, progress=True, num_classes=7)


class DeepLabV3Wrapper(nn.Module):
    def __init__(self, model):
        super(DeepLabV3Wrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["out"]