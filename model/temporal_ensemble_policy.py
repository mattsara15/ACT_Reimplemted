# type: ignore[all]

import torch
import numpy as np

from model.act import ACTModel

class TemporalEnsemblePolicy:
    def __init__(
        self,
        act_model: ACTModel,
        device: torch.device,
        chunk_size: int = 10,
        num_samples: int = 1,
        temporal_ensemble: bool = True,
    ):
        self.act_model = act_model
        self.device = device
        self.chunk_size = chunk_size
        self.num_samples = num_samples
        self.temporal_ensemble = temporal_ensemble

        # Temporal ensemble buffer
        self.action_buffer = []
        self.buffer_size = chunk_size

    def reset(self):
        self.action_buffer = []

    @torch.no_grad
    def get_action(self, image:torch.Tensor, state: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            action_chunk = self.act_model.select_action(image, state, num_samples=self.num_samples)
            action_chunk = action_chunk.cpu().detach().squeeze(0).numpy()  # (chunk_size, action_dim)

            if self.temporal_ensemble:
                # Add to buffer
                self.action_buffer.append(action_chunk)

                # Keep buffer size limited
                if len(self.action_buffer) > self.buffer_size:
                    self.action_buffer.pop(0)

                # Ensemble: average the first action from all chunks in buffer
                # Weight more recent predictions higher
                weights = np.exp(np.linspace(0, 1, len(self.action_buffer)))
                weights = weights / weights.sum()

                ensembled_action = np.zeros(action_chunk.shape[1])
                for i, (chunk, weight) in enumerate(zip(self.action_buffer, weights)):
                    # Use the i-th action from this chunk (accounting for time offset)
                    action_idx = len(self.action_buffer) - 1 - i
                    if action_idx < len(chunk):
                        ensembled_action += weight * chunk[action_idx]

                return ensembled_action
            else:
                # No ensembling: just use first action from chunk
                return action_chunk[0]
