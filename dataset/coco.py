import os
from PIL import Image
from torch.utils.data import Dataset


class COCODataset(Dataset):
    """
    This class loads the COCO dataset(or subset) and returns the image and target.
    Args:
        root: The root directory of the dataset
        target: The target(label) of the dataset
        transform: The transform to be applied to the image
    """

    def __init__(self, root, target, transform=None):
        self.root = root
        self.target = target
        self.path = os.path.join(self.root, self.target)
        self.img = os.listdir(self.path)
        self.transform = transform

    def __len__(self):
        """
        Returns the length of the dataset
        """

        return len(self.img)

    def __getitem__(self, idx):
        """
        Returns the image and target at the given index
        Args:
            idx: The index of the image and target
        """

        img_name = self.img[idx]
        img_path = os.path.join(self.path, img_name)
        img = Image.open(img_path).convert("RGB")
        target = self.target  # return numeric target
        if self.transform is not None:
            img = self.transform(img)
        return img, target
