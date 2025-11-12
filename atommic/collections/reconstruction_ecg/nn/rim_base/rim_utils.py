# coding=utf-8
__author__ = "Dimitris Karkalousos"

import math
import torch
import torch.nn.functional as F
from atommic.collections.common.parts.fft import fft1
from typing import Optional
def log_likelihood_gradient_ecg(
    prediction: torch.Tensor,  # [batch, leads, time] or [batch, 1, leads, time] for 2D conv
    measured_ecg: torch.Tensor,  # [batch, leads, time] or [batch, 1, leads, time] for 2D conv
    mask: torch.Tensor,  # [batch, leads, time] or [batch, 1, leads, time] for 2D conv
    sigma: float,
    update_in_frequency: Optional[bool]= False,
) -> torch.Tensor:
    """
    Compute the gradient of the log-likelihood for ECG lead reconstruction.
    Missing leads are excluded using the mask.
    """
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.item()
    sigma = max(sigma, 1.0)

    # Observed gradient (data consistency)
    mask_gradients = (prediction - measured_ecg) * mask / sigma
    pred_gradients = torch.zeros_like(prediction)
    pred_gradients[..., 2, :] = prediction[..., 2,:] - (prediction[...,1, :] - prediction[...,0, :])
    pred_gradients[..., 3, :] = prediction[...,3,:] - (-(prediction[...,0, :] + prediction[...,1, :]) / 2)
    pred_gradients[..., 4, :] = prediction[...,4,:] - (prediction[...,0, :] - (prediction[...,1, :]/2))
    pred_gradients[..., 5, :] = prediction[...,5,:] - (prediction[...,1, :] - (prediction[...,0, :]/2))
    
    if update_in_frequency:
        mask_gradients = fft1(mask_gradients.squeeze(1), time_dim=-1)
        pred_gradients = fft1(pred_gradients.squeeze(1), time_dim=-1)
        prediction = fft1(prediction.squeeze(1), time_dim=-1)
        gradients = torch.cat([prediction,mask_gradients,pred_gradients], dim= -1).permute(0,3,1,2)
        
    else:
        gradients = torch.cat([prediction,mask_gradients,pred_gradients], dim=1)

    


    # Total gradient
    return gradients