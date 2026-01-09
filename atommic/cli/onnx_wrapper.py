import torch
import torch.nn as nn


class ONNXInferenceWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        masked_waveform: torch.Tensor,
        mask: torch.Tensor,
        sigma: torch.Tensor,
    ):
        outputs = self.model.forward(
            masked_waveform,
            mask,
            sigma=sigma,
        )
        return outputs[-1][-1]
