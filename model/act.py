# model/act.py
# type: ignore[all]
import torch
import torch.nn as nn

from typing import List, Dict

from model.encoder import CVAE, VisionBlock
from model.decoder import Transformer


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


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
    ):
        super().__init__()
        self._input_image_size = input_image_size
        self._input_state_size = input_state_size
        self._action_space = action_space[0]
        self._K = K  # chunk size
        self._device = device

        self._d_model = 512

        self.B = 10

        # learnable components
        self.cvae_encoder = CVAE(
            self._input_state_size,
            self._action_space,
            d_model=self._d_model,
        ).to(self._device)

        self.z_projector = nn.Linear(32, self._d_model).to(self._device)
        self._vision = VisionBlock(self._input_image_size, self._d_model).to(
            self._device
        )
        self._state_projector = nn.Linear(self._action_space, self._d_model).to(
            self._device
        )

        self.query_embed = nn.Embedding(14, self._d_model).to(self._device)
        self._transformer = Transformer(
            d_model=self._d_model,
            nhead=8,
            dim_feedforward=2048,
            num_encoder_layers=4,
            num_decoder_layers=7,
        ).to(self._device)

        self._action_head = nn.Linear(self._d_model, self._action_space).to(
            self._device
        )

        self._optimizer = torch.optim.Adam(
            list(self._transformer.parameters())
            + list(self.z_projector.parameters())
            + list(self.cvae_encoder.parameters())
            + list(self._vision.parameters())
            + list(self._action_head.parameters())
            + list(self._state_projector.parameters())
            + list(self.query_embed.parameters()),
            lr=0.0000625,
        )

    @torch.no_grad
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def train(
        self, image: torch.Tensor, state: torch.Tensor, action: torch.Tensor
    ) -> Dict[str, float]:
        z, z_mean, z_log = self.cvae_encoder.forward(state, action)
        z_proj = self.z_projector(z)

        image_feats, pos_token = self._vision(image)
        projected_joints = self._state_projector(state).squeeze(1)
        hs = self._transformer(
            image_feats,
            None,
            self.query_embed.weight,
            pos_token,
            z_proj,
            projected_joints,
        )  # ,self.additional_pos_embed.weight)[0]
        action_output = self._action_head(hs).squeeze(0).permute(0, 1, 2)
        print(action_output.shape)

        self._optimizer.zero_grad()
        recon_loss = torch.nn.functional.l1_loss(action_output, action)
        kl_loss = kl_divergence(z_mean, z_log)
        loss = recon_loss + self.B * kl_loss
        loss.backward()
        self._optimizer.step()

        return {"train_loss": loss, "recon_lss": recon_loss, "kl_loss": kl_loss}
