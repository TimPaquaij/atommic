# coding=utf-8
__author__ = "Dimitris Karkalousos"

# Taken and adapted from: https://github.com/NVIDIA/NeMo/blob/main/nemo/core/optim/adafactor.py

import math

import torch
from torch.optim.optimizer import Optimizer

__all__ = ["Adafactor"]


class Adafactor(Optimizer):
    """Implements Adafactor algorithm.

    This implementation is based on: `Adafactor: Adaptive Learning Rates with Sublinear Memory Cost`
    (see https://arxiv.org/abs/1804.04235)
    Note that this optimizer internally adjusts the learning rate depending on the *scale_parameter*, *relative_step*
    and *warmup_init* options. To use a manual (external) learning rate schedule you should set `scale_parameter=False`
    and `relative_step=False`.

    Returns
    -------
    Optimizer
        Adafactor Optimizer.
    """

    def __init__(
        self,
        params,
        lr=None,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        scale_parameter=True,
        relative_step=True,
        warmup_init=False,
        min_step=1e-2,
    ):
        """Inits :class:`Adafactor`.

        Parameters
        ----------
        params : iterable
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr : float (optional)
            External learning rate. Default is ``None``.
        eps : tuple (float, float)
            Regularization constants for square gradient and parameter scale respectively.
            Default is ``(1e-30, 1e-3)``.
        clip_threshold : float
            Threshold of root-mean-square of final gradient update. Default is ``1.0``.
        decay_rate : float
            Coefficient used to compute running averages of square gradient. Default is ``-0.8``.
        beta1 : float
            Coefficient used for computing running averages of gradient. Default is ``None``.
        weight_decay : float (optional)
            Weight decay (L2 penalty). Default is ``0``.
        scale_parameter : bool
            If True, learning rate is scaled by root-mean-square of parameter. Default is ``True``.
        relative_step : bool
            If True, time-dependent learning rate is computed instead of external learning rate. Default is ``True``.
        warmup_init : bool
            Time-dependent learning rate computation depends on whether warm-up initialization is being used.
            Default is `False``.
        """
        if lr is not None and relative_step:
            raise ValueError("Cannot combine manual lr and relative_step options")
        if warmup_init and not relative_step:
            raise ValueError("warmup_init requires relative_step=True")
        self.min_step = min_step

        defaults = {
            "lr": lr,
            "eps": eps,
            "clip_threshold": clip_threshold,
            "decay_rate": decay_rate,
            "beta1": beta1,
            "weight_decay": weight_decay,
            "scale_parameter": scale_parameter,
            "relative_step": relative_step,
            "warmup_init": warmup_init,
            "min_step": min_step,
        }
        super().__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        """Whether optimizer supports memory efficient fp16"""
        return True

    @property
    def supports_flat_params(self):
        """Whether the optimizer supports flat parameters."""
        return False

    def _get_lr(self, param_group, param_state):
        """Returns the learning rate for the current layer."""
        rel_step_sz = param_group["lr"]
        if param_group["relative_step"]:
            min_step = 1e-6 * param_state["step"] if param_group["warmup_init"] else self.min_step
            rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state["step"]))
        param_scale = 1.0
        if param_group["scale_parameter"]:
            param_scale = max(param_group["eps"][1], param_state["RMS"])
        return param_scale * rel_step_sz

    def step(self, closure=None):  # noqa: MC0001
        """Performs a single optimization step.

        Parameters
        ----------
        closure : callable (optional)
            A closure that reevaluates the model and returns the loss.
        """
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError("Adafactor does not support sparse gradients.")

                state = self.state[p]
                grad_shape = grad.shape

                factored, use_first_moment = self._get_options(group, grad_shape)
                # State Initialization
                if len(state) == 0:
                    state["step"] = 0

                    if use_first_moment:
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(grad)
                    if factored:
                        state["exp_avg_sq_row"] = torch.zeros(grad_shape[:-1]).to(grad)
                        state["exp_avg_sq_col"] = torch.zeros(grad_shape[:-2] + grad_shape[-1:]).to(grad)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                    state["RMS"] = 0
                else:
                    if use_first_moment:
                        state["exp_avg"] = state["exp_avg"].to(grad)
                    if factored:
                        state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(grad)
                        state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(grad)
                    else:
                        state["exp_avg_sq"] = state["exp_avg_sq"].to(grad)

                p_data_fp32 = p.data
                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p_data_fp32 = p_data_fp32.float()

                state["step"] += 1
                state["RMS"] = self._rms(p_data_fp32)
                group["lr"] = self._get_lr(group, state)

                beta2t = 1.0 - math.pow(state["step"], group["decay_rate"])
                update = (grad**2) + group["eps"][0]
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]

                    exp_avg_sq_row.mul_(beta2t).add_(update.mean(dim=-1), alpha=1.0 - beta2t)
                    exp_avg_sq_col.mul_(beta2t).add_(update.mean(dim=-2), alpha=1.0 - beta2t)

                    # Approximation of exponential moving average of square of gradient
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    update.mul_(grad)
                else:
                    exp_avg_sq = state["exp_avg_sq"]

                    exp_avg_sq.mul_(beta2t).add_(update, alpha=1.0 - beta2t)
                    update = exp_avg_sq.rsqrt().mul_(grad)

                update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))
                update.mul_(group["lr"])

                if use_first_moment:
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(group["beta1"]).add_(update, alpha=1 - group["beta1"])
                    update = exp_avg

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(p_data_fp32, alpha=-group["weight_decay"] * group["lr"])

                p_data_fp32.add_(-update)

                if p.data.dtype in {torch.float16, torch.bfloat16}:
                    p.data.copy_(p_data_fp32)

        return loss

    @staticmethod
    def _get_options(param_group, param_shape):
        """Returns the options for the current layer."""
        factored = len(param_shape) >= 2
        use_first_moment = param_group["beta1"] is not None
        return factored, use_first_moment

    @staticmethod
    def _rms(tensor):
        """Compute the root-mean-square of a tensor."""
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
        """Compute the square of the gradient, but approximate the sqrt using the exponential moving average of the
        squared gradient.
        """
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)
