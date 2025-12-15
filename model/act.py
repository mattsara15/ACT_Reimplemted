# model/act.py
# type: ignore[all]
import torch
import math
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

        self.B = 10

        # learnable components
        self.cvae_encoder = CVAE(
            self._input_state_size,
            self._action_space,
            d_model=self._d_model,
        ).to(self._device)
        self.z_projector = nn.Linear(32, self._d_model).to(self._device)
        
        
        self._vision = VisionBlock(
            self._input_image_size, self._d_model, self._device
        ).to(self._device)
        
        self._state_projector = nn.Linear(self._action_space, self._d_model).to(
            self._device
        )

        self.pos_encoder = PositionalEncoding(self._d_model, max_len=self._K, dropout=0.1).to(
            self._device
        )
        self.action_queries = nn.Parameter(torch.randn(self._K, 1, self._d_model)).to(
            self._device
        )

        self._encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                self._d_model,
                nhead=8,
                activation="relu",
                dim_feedforward=dim_feedforward,
            ),
            num_encoder_layers,
        ).to(self._device)

        self._decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                self._d_model,
                nhead=8,
                activation="relu",
                dim_feedforward=dim_feedforward,
            ),
            num_decoder_layers,
            nn.LayerNorm(self._d_model),
        ).to(self._device)

        self._action_head = nn.Linear(self._d_model, self._action_space).to(
            self._device
        )

        self._optimizer = torch.optim.Adam(
            list(self._encoder.parameters())
            + list(self._decoder.parameters())
            + list(self.z_projector.parameters())
            + list(self.cvae_encoder.parameters())
            + list(self._vision.parameters())
            + list(self._action_head.parameters())
            + list(self.pos_encoder.parameters())
            + list(self._state_projector.parameters()),
            lr=0.0000625,
        )

    def select_action(
        visual_features: torch.Tensor, state: torch.Tensor, num_samples: int = 1
    ) -> torch.Tensor:
        with torch.no_grad():
            if num_samples == 1:
                actions, _, _ = self.forward(
                    visual_features, state, actions=None, sample_latent=True
                )
                return actions
            else:
                # Sample multiple times and average
                action_samples = []
                for _ in range(num_samples):
                    actions, _, _ = self.forward(
                        visual_features, state, actions=None, sample_latent=True
                    )
                    action_samples.append(actions)

                return torch.stack(action_samples).mean(dim=0)

    def forward(
        self, image: torch.Tensor, state: torch.Tensor, latent: Optional[torch.Tensor]
    ) -> torch.Tensor:
        B, _, _, _ = image.shape

        z = latent
        if not latent:
            # provide empty encoder at inference time
            z = torch.zeros(B, 32, device=self._device)

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
        action_output = self._action_head(decoder_output).squeeze(0).permute(1, 0, 2)
        return action_output

    def train(
        self, image: torch.Tensor, state: torch.Tensor, action: torch.Tensor
    ) -> Dict[str, float]:
        # run auto-encoder
        z, z_mean, z_log = self.cvae_encoder.forward(state, action)

        action_output = self.forward(image, state, z)

        # optimize
        self._optimizer.zero_grad()
        recon_loss = torch.nn.functional.l1_loss(action_output, action)
        total_kld = kl_divergence(z_mean, z_log)
        loss = recon_loss + self.B * total_kld
        loss.backward()
        self._optimizer.step()

        return {"train_loss": loss, "recon_lss": recon_loss, "kl_loss": total_kld}
