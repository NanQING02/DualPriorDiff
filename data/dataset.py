"""Datasets for image inpainting.

Masked images are produced beforehand (see ``data/make_masks.py``) and stored as
ordinary image files. Ground-truth and masked folders must contain the same file
names; pairs are matched in sorted file-name order.
"""

import os

import torchvision
from PIL import Image
from torch.utils.data import Dataset

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

MIN_MAX = (-1, 1)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_paths_from_images(path):
    """Sorted list of image paths below ``path``."""
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                images.append(os.path.join(dirpath, fname))
    assert images, '{:s} has no valid image file'.format(path)
    return sorted(images)


def check_pairs(*lists):
    names = [os.path.basename(p) for p in lists[0]]
    for other in lists[1:]:
        assert len(other) == len(lists[0]), \
            'folders contain different numbers of images (%d vs %d)' % (len(lists[0]), len(other))
        assert [os.path.basename(p) for p in other] == names, \
            'folders contain different file names'


def to_tensor(img, split='val'):
    t = torchvision.transforms.ToTensor()(img)
    if split == 'train':
        t = torchvision.transforms.RandomHorizontalFlip()(t)
    return t * (MIN_MAX[1] - MIN_MAX[0]) + MIN_MAX[0]


class PairedMaskedDataset(Dataset):
    """({ground truth, masked input}, file name) for training and validation."""

    def __init__(self, gt_dir, masked_dir, split='train'):
        self.gt_paths = get_paths_from_images(gt_dir)
        self.masked_paths = get_paths_from_images(masked_dir)
        check_pairs(self.gt_paths, self.masked_paths)
        self.split = split

    def __len__(self):
        return len(self.gt_paths)

    def __getitem__(self, index):
        gt = Image.open(self.gt_paths[index]).convert('RGB')
        masked = Image.open(self.masked_paths[index]).convert('RGB')
        name = os.path.splitext(os.path.basename(self.gt_paths[index]))[0]
        return {'HR': to_tensor(gt, self.split), 'MOS': to_tensor(masked, self.split)}, name


class MaskedDataset(Dataset):
    """Masked input only, used when the results are saved at alternating epochs."""

    def __init__(self, masked_dir, limit=-1):
        self.paths = get_paths_from_images(masked_dir)
        self.limit = limit

    def __len__(self):
        return len(self.paths) if self.limit == -1 else min(self.limit, len(self.paths))

    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert('RGB')
        name = os.path.splitext(os.path.basename(self.paths[index]))[0]
        return to_tensor(img, 'val'), name


class MaskedInferenceDataset(Dataset):
    """Masked input for inference; returns the file name together with the tensor."""

    def __init__(self, masked_dir, limit=-1):
        self.paths = get_paths_from_images(masked_dir)
        self.limit = limit

    def __len__(self):
        return len(self.paths) if self.limit == -1 else min(self.limit, len(self.paths))

    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert('RGB')
        name = os.path.basename(self.paths[index])
        return to_tensor(img, 'val'), name
