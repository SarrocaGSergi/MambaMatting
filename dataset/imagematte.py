import os
import random
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from .augmentation import MotionAugmentation


class ImageMatteDataset(Dataset):
    def __init__(self, imagematte_dir, background_image_dir, size, transform):
        self.imagematte_dir = imagematte_dir
        self.imagematte_files = os.listdir(os.path.join(imagematte_dir, 'fgr'))
        self.background_image_dir = background_image_dir
        self.background_image_files = os.listdir(background_image_dir)
        self.size = size
        self.transform = transform

    def __len__(self):
        return max(len(self.imagematte_files), len(self.background_image_files))

    def __getitem__(self, idx):
        bgrs = self._get_static_bg()
        fgrs, phas = self._get_static_image(idx)

        if self.transform is not None:
            return self.transform(fgrs, phas, bgrs)
        else:
            to_tensor = T.Compose([T.Resize((1080, 1920)), T.CenterCrop((1080, 1920)), T.ToTensor()])
            fgrs = to_tensor(fgrs)
            phas = to_tensor(phas)
            bgrs = to_tensor(bgrs)
            return fgrs, phas, bgrs

    def _get_static_image(self, idx):
        with Image.open(os.path.join(self.imagematte_dir, 'fgr',
                                     self.imagematte_files[idx % len(self.imagematte_files)])) as fgr, \
                Image.open(os.path.join(self.imagematte_dir, 'pha',
                                        self.imagematte_files[idx % len(self.imagematte_files)])) as pha:
            fgr = self._downsample_if_needed(fgr.convert('RGB'))
            pha = self._downsample_if_needed(pha.convert('L'))
        return fgr, pha

    def _get_static_bg(self):
        with Image.open(os.path.join(self.background_image_dir, self.background_image_files[
            random.choice(range(len(self.background_image_files)))])) as bgr:
            bgr = self._downsample_if_needed(bgr.convert('RGB'))
        return bgr

    def _downsample_if_needed(self, img):
        w, h = img.size
        if min(w, h) > self.size:
            scale = self.size / min(w, h)
            w = int(scale * w)
            h = int(scale * h)
            img = img.resize((w, h))
        return img
