from torchvision.transforms import functional as F


class HorizontalFlipTTA:
    def __init__(self):
        pass

    @staticmethod
    def augment_image(image):
        image = F.hflip(image)
        return image

    @staticmethod
    def deaugment_mask(mask):
        mask = F.hflip(mask)
        return mask
