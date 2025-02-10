from torchvision.transforms.functional import resize, InterpolationMode
import torch

def resize_max_edge(
    image,
    size: int,
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    antialias: bool = True,
):
    height, width = image.shape[-2:]
    scale = min(size / height, size / width)
    new_height = round(scale * height)
    new_width = round(scale * width)
    return resize(
        image,
        size=(new_height, new_width),
        interpolation=interpolation,
        antialias=antialias,
    )


class Resize(torch.nn.Module):
    def __init__(self, 
                 size: int = 256):
        super().__init__()
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return resize(
            x, self.size, interpolation=InterpolationMode.BILINEAR, antialias=True
        )