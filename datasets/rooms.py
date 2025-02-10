import json
from typing import Sequence, Tuple, Union
import numpy as np

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import io
from torchvision.transforms.functional import center_crop, hflip
from kornia.geometry.transform import warp_perspective

from utils.resize import resize_max_edge


class Rooms(Dataset):
    def __init__(self,
                 dataframe_path: Union[str, Sequence[str]],
                 max_edge: int,
                 augment: bool = False,
                 normalize: bool = False,
                 mean: Sequence[float] = (0.5, 0.5, 0.5),
                 std: Sequence[float] = (0.5, 0.5, 0.5),
                 seed: int = None
                 ):
        super().__init__()
        if isinstance(dataframe_path, str):
            self.dataframe = pd.read_csv(dataframe_path)
        else:
            self.dataframe = pd.concat([pd.read_csv(path) for path in dataframe_path])

        self.max_edge = max_edge
        self.augment = augment
        self.normalize = normalize
        self.mean, self.std = torch.tensor(mean).view(1, 3, 1, 1), torch.tensor(std).view(1, 3, 1, 1)

        self.seed = seed
        self.reset_rng()
    
    def reset_rng(self):
        self.rng = np.random.default_rng(self.seed)

    def __len__(self) -> int:
        return len(self.dataframe)
    
    def load_item(self, idx: int):
        row = self.dataframe.iloc[idx]
        image = io.read_image(row['DONE'], io.ImageReadMode.RGB).unsqueeze(0) / 255.0
        return image
    
    def normalization(self, image: torch.Tensor) -> torch.Tensor:
        d = image.device
        return (image - self.mean.to(d)) / self.std.to(d)
    
    def denormalization(self, image: torch.Tensor) -> torch.Tensor:
        d = image.device
        return image * self.std.to(d) + self.mean.to(d)

    def __getitem__(self, idx: int):
        image = self.load_item(idx)
        image = center_crop(image, min(image.shape[-2:]))
        image = resize_max_edge(image, self.max_edge)

        if self.augment:
            if self.rng.random() < 0.5:
                image = hflip(image)
        
        if self.normalize:
            image = self.normalization(image)
        

        image = image.reshape(3, *image.shape[-2:]).contiguous()
        noise = torch.zeros_like(image) # Placeholder
        return noise, image

if __name__ == '__main__':
    from utils.save import save_image

    max_edge = 512
    dataset = Rooms(dataframe_path='data/interior/ps-straighten-ok-train-interior.csv',
                             max_edge=max_edge,
                             augment=True,
                             normalize=False)
    _, image = dataset[205]

    save_image(image, 'image.png')
    