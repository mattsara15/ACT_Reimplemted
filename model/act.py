# model/act.py
# type: ignore[all]

import os

import torch
import math
import numpy as np
import torch.nn as nn

from typing import List, Dict, Optional

from model.cvae import CVAE
from model.vision_block import VisionBlock


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch, d_model)
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    return total_kld


# Implement action chunking transformer model here
class ACTModel(nn.Module):
    def __init__(
        self,
        input_image_size: List[int],
        input_state_size: int,
        action_space: int,
        K: int,
        device: torch.device,
        d_model=512,
        num_encoder_layers=4,
        num_decoder_layers=7,
        dim_feedforward=2048,
    ):
        super().__init__()
        self._input_image_size = input_image_size
        self._input_state_size = input_state_size
        self._action_space = action_space[0]
        self._K = K  # chunk size
        self._device = device

        self._d_model = d_model

        # learnable components
        self.cvae_encoder = CVAE(
            self._input_state_size,
            self._action_space,
            d_model=self._d_model,
        )
        self.z_projector = nn.Linear(32, self._d_model)

        self._vision = VisionBlock(
            self._input_image_size, self._d_model, self._device
        )

        self._state_projector = nn.Linear(self._action_space, self._d_model)

        self.pos_encoder = PositionalEncoding(
            self._d_model, max_len=self._K, dropout=0.1
        )
        self.action_queries = nn.Parameter(torch.randn(self._K, 1, self._d_model))

        self._encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                self._d_model,
                nhead=8,
                activation="relu",
                dim_feedforward=dim_feedforward,
            ),
            num_encoder_layers,
        )

        self._decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                self._d_model,
                nhead=8,
                activation="relu",
                dim_feedforward=dim_feedforward,
            ),
            num_decoder_layers,
            nn.LayerNorm(self._d_model),
        )

        self._action_head = nn.Linear(self._d_model, self._action_space)

    def forward(
        self, image: torch.Tensor, state: torch.Tensor, inference_latent: bool, actions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, _, _, _ = image.shape

        z = torch.zeros(B, 32, device=self._device)
        z_mean = None
        z_log = None
        if inference_latent:
            assert actions is not None, "Actions must be provided when inference_latent is True"
            z, z_mean, z_log = self.cvae_encoder.forward(state, actions)

        z_proj = self.z_projector(z)

        image_feats, _ = self._vision(image)
        projected_joints = self._state_projector(state).squeeze(1)

        # encoder
        encoder_input = torch.stack(
            [image_feats, projected_joints, z_proj], dim=0
        )  # pos_token optional
        memory = self._encoder(encoder_input)

        # decoder
        action_queries = self.action_queries.expand(-1, B, -1)
        action_queries = self.pos_encoder(action_queries)

        # Transformer decoder
        decoder_output = self._decoder(
            action_queries, memory
        )  # (chunk_size, batch, hidden_dim)
        predicted_action = self._action_head(decoder_output).squeeze(0).permute(1, 0, 2)
        return predicted_action, z_mean, z_log


class ACTModelWrapper(nn.Module):
    def __init__(self, model: ACTModel, enhanced_debug: bool = True):
        super().__init__()

        self.B = 10

        self._enhanced_debug : bool = enhanced_debug

        self.model = model
        self._optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-4,
        )

    def select_action(
        self, visual_features: torch.Tensor, state: torch.Tensor, num_samples: int = 1
    ) -> torch.Tensor:
        with torch.no_grad():
            if num_samples == 1:
                actions, _, _ = self.forward(visual_features, state, inference_latent=False)
                return actions.squeeze(0)
            else:
                # Sample multiple times and average
                action_samples = []
                for _ in range(num_samples):
                    actions, _, _ = self.forward(visual_features, state, inference_latent=False)
                    action_samples.append(actions)

                return torch.stack(action_samples).mean(dim=0)
            
    def train(
        self, image: torch.Tensor, state: torch.Tensor, actions: torch.Tensor
    ) -> Dict[str, float]:
        action_output, z_mean, z_log = self.model(image, state, actions=actions, inference_latent=True)

        # optimize
        self._optimizer.zero_grad()
        recon_loss = torch.nn.functional.l1_loss(action_output, actions)
        total_kld = kl_divergence(z_mean, z_log)
        loss = recon_loss + self.B * total_kld
        loss.backward()
        self._optimizer.step()

        # Introspect the quality of action predictions
        predicted_actions_x = []
        predicted_actions_y = []
        gt_actions_x = []
        gt_actions_y = []
        
        if self._enhanced_debug:
            for action in action_output:
                predicted_actions_x.append(action[0][0].cpu().detach())
                predicted_actions_y.append(action[0][1].cpu().detach())

            for action in actions:
                gt_actions_x.append(action[0][0].cpu().detach())
                gt_actions_y.append(action[0][1].cpu().detach())

        return {
            "train_loss": loss,
            "recon_lss": recon_loss,
            "kl_loss": total_kld,
            "predicted_action_histogram_x": np.asarray(predicted_actions_x),
            "predicted_action_histogram_y": np.asarray(predicted_actions_y),
            "gt_action_histogram_x": np.asarray(gt_actions_x),
            "gt_action_histogram_y": np.asarray(gt_actions_y),
        }

    def save_checkpoint(self, path: str) -> bool:
        parent = os.path.dirname(path)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

        checkpoint = {
            "model_state": self.model.state_dict(),
        }
        torch.save(checkpoint, path)
        return True

    def load_from_checkpoint(self, path: str) -> bool:
        checkpoint = torch.load(path, map_location=self._device)
        if "model_state" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state"])
        return True
    