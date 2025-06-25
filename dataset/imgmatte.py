from __future__ import annotations

import os
import random
import torchvision
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import torchvision.transforms.functional as TF


class ImageMatte(Dataset):
    def __init__(self, imagematte_dir: str | Path, transform: torchvision.transforms):
        self.imagematte_dir = imagematte_dir
        self.pha_dir = os.path.join(imagematte_dir, 'pha')
        self.img_dir = os.path.join(imagematte_dir, 'image')
        self.images = os.listdir(self.img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = Image.open(os.path.join(self.img_dir, self.images[idx]))
        gt_image = Image.open(os.path.join(self.pha_dir, self.images[idx]))

        if self.transform:
            seed = random.randint(0, 2**32)
            random.seed(seed)
            image = self.transform(image)
            random.seed(seed)
            gt_image = self.transform(gt_image)

        else:
            image = TF.to_tensor(image)
            gt_image = TF.to_tensor(gt_image)

        return image, gt_image


