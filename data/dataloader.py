from glob import glob
from PIL import Image
from typing import Callable, Optional
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
import torch
import numpy as np
import os
import yaml

__DATASET__ = {}

def register_dataset(name: str):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATASET__[name] = cls
        return cls
    return wrapper


def get_dataset(name: str, root: str, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Dataset {name} is not defined.")
    return __DATASET__[name](root=root, **kwargs)


def get_dataloader(dataset: VisionDataset,
                   batch_size: int, 
                   num_workers: int, 
                   train: bool):
    dataloader = DataLoader(dataset, 
                            batch_size, 
                            shuffle=train, 
                            num_workers=num_workers, 
                            drop_last=train)
    return dataloader


@register_dataset(name='ffhq')
class FFHQDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None):
        super().__init__(root, transforms)

        self.fpaths = sorted(glob(root + '/**/*.png', recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img


@register_dataset(name='func')
class FuncDataset:
    def __init__(self, root:str, normalize, transforms: Optional[Callable]=None):
        folder = root
        files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.npy')]
        self.f_value = []
        self.normalize = normalize
        for file in files:
            self.f_value.append(np.load(file))
            

    def __len__(self):
        return len(self.f_value)

    def __getitem__(self, index: int):
        f_value = torch.tensor(self.f_value[index])
        
        if self.normalize != 0:
            f_max = torch.max(f_value)
            f_min = torch.min(f_value)
            f_value = 2 * (f_value - f_min)/(f_max - f_min) - 1
        return f_value
