# model/act.py
# type: ignore[all]

import math
import torch
import torch.nn as nn

from torchvision.models import resnet18, ResNet18_Weights
from torch.autograd import Variable

from typing import List

class PositionEmbedding_Sine_2D(nn.Module):
    def __init__(
        self, num_pos_feats=64, temperature=10000, normalize=False, scale=None
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x: torch.Tensor):
        not_mask = torch.ones_like(x[0, [0]])
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos



class VisionBlock(nn.Module):
    def __init__(
        self,
        input_image_size: List[int],
        d_model,
        device
    ):
        super().__init__()
        self._input_image_size = input_image_size
        self._d_model = d_model
        self._device = device

        self.resnet18 = resnet18(weights=ResNet18_Weights.DEFAULT, progress=True)
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-1])
        print(self.resnet18)

        self.position_embedding = PositionEmbedding_Sine_2D(self._d_model)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        B, _, _, _ = image.shape

        # Apply ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=self._device)
        std = torch.tensor([0.229, 0.224, 0.225], device=self._device)
        img = (image - mean.view(1, 3, 1, 1)) / std.view(1, 3, 1, 1)

        y = self.resnet18(img)
        y_flat = torch.flatten(y, start_dim=2).squeeze(-1)
        return y_flat, None
