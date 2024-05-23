from typing import List, Optional

import matplotlib.pyplot as plt
import torch
from model import AllTnn
from PIL import Image as I
from timm.data.transforms_factory import create_transform


def get_images(image_paths: List[str]) -> List[I.Image]:
    out = []
    for path in image_paths:
        image = I.open(path)
        out.append(image)
    return out


def process_input_batch(
    images_batch: List[I.Image],
    input_size: int = 128,
    min_scale: float = 0.4,
    hflip: float = 0.5,
    color_jitter: Optional[float] = 0.4,
    interpolation: str = "bicubic",
) -> torch.Tensor:

    image_transform = create_transform(
        input_size=input_size,
        is_training=False,
        scale=(min_scale, 1.0),
        hflip=hflip,
        color_jitter=None if color_jitter == 0 else color_jitter,
        interpolation=interpolation,
    )
    transformed_images = [image_transform(img.convert("RGB")) for img in images_batch]

    return torch.stack(transformed_images)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(
    "checkpoints/epoch_6_iter_494_val_loss_-5.46_2024-05-22_21:30:56.240620_imagenet_100.pt",
    map_location=device,
)

model_cfg = checkpoint["model_config"]
model_cfg.dropout_p = 0
model = AllTnn(model_cfg, device).to(device)

state_dict = checkpoint["model"]
unwanted_prefix = "_orig_mod."
for k, _ in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
model.load_state_dict(state_dict)


image_paths = [
    "data/test_images/horizontal.png",
    "data/test_images/left_diagonal.png",
    "data/test_images/right_diagonal.png",
    "data/test_images/vertical.png",
]

pil_images = get_images(image_paths)
inputs = process_input_batch(pil_images)

_, _, _, all_activations = model(inputs)

all_activations = [act.detach().numpy() / 100 for act in all_activations]

num_layers = len(all_activations)

fig, axes = plt.subplots(4, 6, figsize=(5, 5))

for i, activation in enumerate(all_activations):
    ax = axes.flat[i]

    act_min, act_max = activation.min(), activation.max()
    norm_activation = (activation - act_min) / (act_max - act_min + 1e-8)

    norm_activation = norm_activation[0, 0, :, :]
    ax.imshow(norm_activation, cmap="viridis")
    ax.axis("off")
    ax.set_title(f"Horizontal, layer {i+1}")

for i, activation in enumerate(all_activations):
    ax = axes.flat[i + 6]

    act_min, act_max = activation.min(), activation.max()
    norm_activation = (activation - act_min) / (act_max - act_min + 1e-8)

    norm_activation = norm_activation[1, 0, :, :]
    ax.imshow(norm_activation, cmap="viridis")
    ax.axis("off")
    ax.set_title(f"Left diagonal, layer {i+1}")

for i, activation in enumerate(all_activations):
    ax = axes.flat[i + 12]

    act_min, act_max = activation.min(), activation.max()
    norm_activation = (activation - act_min) / (act_max - act_min + 1e-8)

    norm_activation = norm_activation[2, 0, :, :]
    ax.imshow(norm_activation, cmap="viridis")
    ax.axis("off")
    ax.set_title(f"Right diagonal, layer {i+1}")

for i, activation in enumerate(all_activations):
    ax = axes.flat[i + 18]

    act_min, act_max = activation.min(), activation.max()
    norm_activation = (activation - act_min) / (act_max - act_min + 1e-8)

    norm_activation = norm_activation[3, 0, :, :]
    ax.imshow(norm_activation, cmap="viridis")
    ax.axis("off")
    ax.set_title(f"Vertical, layer {i+1}")


plt.tight_layout()
plt.show()
