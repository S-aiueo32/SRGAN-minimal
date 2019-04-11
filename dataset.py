from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop
from torchvision.transforms import functional as F

from PIL import Image, ImageOps
from pathlib import Path
import random


def padding(img, scale):
    width, height = img.size
    pad_h = width % scale
    pad_v = height % scale
    img = F.pad(img, (0, 0, scale - pad_h, scale - pad_v), padding_mode='edge')
    return img


class DatasetFromFolder(Dataset):
    def __init__(self, image_dir, patch_size, upscale_factor, data_augmentation=True):
        super(DatasetFromFolder, self).__init__()
        self.filenames = [f for f in Path(image_dir).glob('*.jpg')]
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.data_augmentation = data_augmentation
        self.crop = RandomCrop(self.patch_size)

    def __getitem__(self, index):
        target_img = Image.open(self.filenames[index]).convert('RGB')
        target_img = self.crop(target_img)

        if self.data_augmentation:
            if random.random() < 0.5:
                target_img = ImageOps.flip(target_img)
            if random.random() < 0.5:
                target_img = ImageOps.mirror(target_img)
            if random.random() < 0.5:
                target_img = target_img.rotate(180)

        down_size = (self.patch_size // self.upscale_factor,) * 2
        input_img = target_img.resize(down_size, Image.BICUBIC)

        return F.to_tensor(input_img), F.to_tensor(target_img) * 2 - 1

    def __len__(self):
        return len(self.filenames)


class DatasetFromFolderEval(Dataset):
    def __init__(self, image_dir, upscale_factor):
        super(DatasetFromFolderEval, self).__init__()
        self.filenames = [f for f in Path(image_dir).glob('*.jpg')]
        self.upscale_factor = upscale_factor

    def __getitem__(self, index):
        target_img = Image.open(self.filenames[index]).convert('RGB')
        target_img = padding(target_img, self.upscale_factor)

        down_size = (target_img.size[0] // self.upscale_factor,
                     target_img.size[1] // self.upscale_factor)
        input_img = target_img.resize(down_size, Image.BICUBIC)

        return F.to_tensor(input_img), F.to_tensor(target_img), self.filenames[index].stem

    def __len__(self):
        return len(self.filenames)
