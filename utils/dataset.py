import os

from PIL import Image, ImageOps
from torch.utils.data import Dataset as BaseDataset
from utils.general import Augmentation


class DamageDataset(BaseDataset):
    """DamageDataset:

    Args:
        root: data path to images folder
        image_size: expected input image size
        transforms: default transformer uses (Augmentation)
    """

    def __init__(
            self,
            root: str,
            image_size: int = 1024,
            transforms: Augmentation = Augmentation(),
            target_transforms=None,
    ) -> None:
        self.root = root
        self.image_size = image_size
        self.filenames = [os.path.splitext(filename)[0] for filename in os.listdir(os.path.join(self.root, "images"))]
        if not self.filenames:
            raise FileNotFoundError(f"Files not found in {root}")

        self.transforms = transforms
        self.target_transforms = target_transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]

        # image path
        image_path = os.path.join(self.root, f"images{os.sep}{filename}.jpg")
        mask_path = os.path.join(self.root, f"masks{os.sep}{filename}.png")

        # image load
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        # resize
        image, mask = self.resize_pil(image, mask, image_size=self.image_size)

        assert image.size == mask.size, f"`image`: {image.size} and `mask`: {mask.size} are not the same"

        if self.transforms is not None:
            image, mask = self.transforms(image, mask)

        if self.target_transforms is not None:
            image, mask = self.target_transforms(image, mask)

        return image, mask

    @staticmethod
    def resize_pil(image, mask, image_size):
        """Letter box resizing: Downscales larges side of the image to a given
        `image_size` then smaller size will be padded by both sides

        Args:
            image: input image
            mask: input mask
            image_size: desired image size
        Returns:
            resized pil image and mask
        """
        w, h = image.size
        scale = min(image_size / w, image_size / h)

        # resize image
        image = image.resize((int(w * scale), int(h * scale)), resample=Image.BICUBIC)
        mask = mask.resize((int(w * scale), int(h * scale)), resample=Image.NEAREST)

        # pad size
        delta_w = image_size - int(w * scale)
        delta_h = image_size - int(h * scale)
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        # pad image
        image = ImageOps.expand(image, (left, top, right, bottom))
        mask = ImageOps.expand(mask, (left, top, right, bottom))

        return image, mask
