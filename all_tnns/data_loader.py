from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from datasets import DatasetDict, load_dataset
from PIL import Image as I
from timm.data.transforms_factory import create_transform


def create_dataset(
    dataset_name: str,
    input_size: int = 128,
    min_scale: float = 0.4,
    hflip: float = 0.5,
    color_jitter: Optional[float] = 0.4,
    interpolation: str = "bicubic",
    keep_in_memory: bool = False,
) -> DatasetDict:
    dsets: DatasetDict = load_dataset(dataset_name, keep_in_memory=keep_in_memory)

    for split, ds in dsets.items():
        image_transform = create_transform(
            input_size=input_size,
            is_training=split == "train",
            scale=(min_scale, 1.0),
            hflip=hflip,
            color_jitter=None if color_jitter == 0 else color_jitter,
            interpolation=interpolation,
        )
        transform = _get_batch_transform(image_transform)
        ds.set_transform(transform)
    return dsets


def _transform(
    batch: Dict[str, List[Union[I.Image, Any]]],
    image_transform: Callable[[I.Image], torch.Tensor],
) -> Dict[str, torch.Tensor]:
    batch["image"] = [image_transform(img.convert("RGB")) for img in batch["image"]]
    return batch


def _get_batch_transform(image_transform: Callable[[I.Image], torch.Tensor]):
    return partial(_transform, image_transform=image_transform)
