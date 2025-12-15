# model/act.py
# type: ignore[all]

import math
import torch
import torch.nn as nn

from torchvision.models import resnet18, ResNet18_Weights
from torch.autograd import Variable

from typing import List


class PositionEmbedding_Sine_1D(nn.Module):
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

    def forward(self, tensor):
        x = tensor
        not_mask = torch.ones_like(x[:, :, 0])
        pos_embed = not_mask.cumsum(1, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            pos_embed = pos_embed / (pos_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos = pos_embed[:, :, None] / dim_t

        pos = torch.stack(
            (pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        return pos

class CVAE(nn.Module):
    def __init__(
        self,
        input_state_size: int,
        action_space: int,
        d_model,
    ):
        super().__init__()
        self._input_state_size = input_state_size
        self._action_space = action_space
        self._d_model = d_model

        self.position_embedding_1d = PositionEmbedding_Sine_1D(self._d_model)

        # learnable components
        self.cls_embed = nn.Embedding(1, self._d_model)
        self._action_projector = nn.Linear(self._action_space, self._d_model)
        self._joint_projector = nn.Linear(self._input_state_size, self._d_model)
        self._encoder = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=self._d_model, nhead=8),
            nn.TransformerEncoderLayer(d_model=self._d_model, nhead=8),
            nn.TransformerEncoderLayer(d_model=self._d_model, nhead=8),
            nn.TransformerEncoderLayer(d_model=self._d_model, nhead=8),
        )
        self._final_linear = nn.Linear(self._d_model, 64)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        B, _, _ = state.shape

        cls_embed = self.cls_embed.weight
        cls_embed = cls_embed.unsqueeze(0).repeat(B, 1, 1)

        projected_joints = self._joint_projector(state)
        projected_action = self._action_projector(action)

        encoder_input = torch.cat(
            [cls_embed, projected_joints, projected_action], dim=1
        )
        pos_embed = self.position_embedding_1d(encoder_input)

        # Run Transformer
        encoder_input = encoder_input.permute(1, 0, 2)
        pos_embed = pos_embed.permute(1, 0, 2)
        encoder_output = self._encoder.forward(encoder_input)

        final_output = self._final_linear(encoder_output)
        final_output = final_output[0]
        z_mean = final_output[:, :32]
        z_logvar = final_output[:, 32:]

        # reparameterization trick to get z
        z_std = z_logvar.div(2).exp()
        eps = Variable(z_std.data.new(z_std.size()).normal_())
        return z_mean + z_std * eps, z_mean, z_logvar


class VisionBlock(nn.Module):
    def __init__(
        self,
        input_image_size: List[int],
        d_model,
    ):
        super().__init__()
        self._input_image_size = input_image_size
        self._d_model = d_model

        weights = ResNet18_Weights.DEFAULT  # TODO - explore pre-init
        self.resnet18 = resnet18(weights=weights, progress=True)
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:6])
        print(self.resnet18)

        self.position_embedding = PositionEmbedding_Sine_1D(self._d_model)

        self._linear = nn.Linear(144, self._d_model)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        B, _, _, _ = image.shape

        y = self.resnet18(image)
        y_flat = torch.flatten(y, start_dim=2)

        result = self._linear(nn.functional.relu(y_flat)) 
        pos = self.position_embedding(result)
        return result, pos
