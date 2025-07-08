"""
Expected directory format:

VideoMatte Train/Valid:
    ├──fgr/
      ├── 0001/
        ├── 00000.jpg
        ├── 00001.jpg
    ├── pha/
      ├── 0001/
        ├── 00000.jpg
        ├── 00001.jpg
        
ImageMatte Train/Valid:
    ├── fgr/
      ├── sample1.jpg
      ├── sample2.jpg
    ├── pha/
      ├── sample1.jpg
      ├── sample2.jpg

Background Image Train/Valid
    ├── sample1.png
    ├── sample2.png

Background Video Train/Valid
    ├── 0000/
      ├── 0000.jpg/
      ├── 0001.jpg/

"""

from pathlib import Path

_ROOT_ = Path("/mnt/nvme0n1/Datasets")

DATA_PATHS = {

    'videomatte': {
        'train': _ROOT_ / 'matting-data/VideoMatte240K_JPEG_SD/train',
        'valid': _ROOT_ / 'matting-data/VideoMatte240K_JPEG_SD/val',
    },

    'videomatteHD': {
        'train': _ROOT_ / 'matting-data/VideoMatte240K_JPEG_HD/train',
        'valid': _ROOT_ / 'matting-data/VideoMatte240K_JPEG_HD/val',
    },

    'brainstorm': {
        'train': _ROOT_ / 'matting-data/Brainstorm/train',
        'valid': _ROOT_ / 'matting-data/Brainstorm/val',
    },

    'brainstorm_bgs':{
        'train': _ROOT_ / 'matting-data/Brainstorm/backgrounds/train',
        'valid': _ROOT_ / 'matting-data/Brainstorm/backgrounds/val'
    },

    'imagematte': {
        'train':  _ROOT_ / 'matting-data/P3M-10k/train',
        'valid':  _ROOT_ / 'matting-data/P3M-10k/val',
    },

    'am2k': {
        'train':  _ROOT_ / 'AM2K/train',
        'valid':  _ROOT_ / 'AM2K/val',
    },

    'bg20k':{
        'train': _ROOT_ / 'BG20K/train',
        'valid': _ROOT_ / 'BG20K/testval'
    },

    'aim': {
        'train':  _ROOT_ / 'matting-data/image-matte/AIM/train',
        'valid':  _ROOT_ / 'matting-data/image-matte/AIM/val',
    },
    'h646': {
        'train':  _ROOT_ / 'matting-data/image-matte/H646/train',
        'valid':  _ROOT_ / 'matting-data/image-matte/H646/val',
    },
    'p3m': {
        'train':  _ROOT_ / 'matting-data/image-matte/P3M-10k/train',
        'valid':  _ROOT_ / 'matting-data/image-matte/P3M-10k/val',
    },

    'background_images': {
        'train':  _ROOT_ / 'matting-data/ImageMatte/train/bgr',
        'valid':  _ROOT_ / 'matting-data/ImageMatte/val/bgr',
    },

    'background_videos': {
        'train':  _ROOT_ / 'matting-data/BackgroundVideos/train',
        'valid':  _ROOT_ / 'matting-data/BackgroundVideos/test',
    },

    'coco_panoptic': {
        'imgdir':  _ROOT_ / 'matting-data/coco/train2017/',
        'anndir':  _ROOT_ / 'matting-data/coco/annotations/panoptic_train2017/',
        'annfile': _ROOT_ / 'matting-data/coco/annotations/panoptic_train2017.json',
    },

    'spd': {
        'imgdir':  _ROOT_ / 'matting-data/SuperviselyPersonDataset/img',
        'segdir':  _ROOT_ / 'matting-data/SuperviselyPersonDataset/seg',
    },

    'youtubevis': {
        'videodir': _ROOT_ / 'matting-data/VOS/train/JPEGImages',
        'annfile':  _ROOT_ / 'matting-data/VOS/train/instances.json',
    }

}
