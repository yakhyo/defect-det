import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

class DeepLabV3Wrapper(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(DeepLabV3Wrapper, self).__init__()
        self.model = deeplabv3_resnet50(pretrained=pretrained, progress=True)
        self.model.classifier = DeepLabHead(2048, num_classes)
        self.in_channels = 3
        self.out_channels = num_classes


    def forward(self, x):
        return self.model(x)["out"]
