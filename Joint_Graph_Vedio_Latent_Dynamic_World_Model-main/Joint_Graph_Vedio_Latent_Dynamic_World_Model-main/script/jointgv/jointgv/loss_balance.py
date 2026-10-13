from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn


class GradientRatioOTController:
    """Balance OT against the backbone objective by shared gradient norms.

    The controller treats ``weight`` as the full-strength OT coefficient.
    The training loop applies the existing epoch warm-up separately.  Only
    parameters reached by *both* objectives enter the two norms, so private
    backbone or graph-head parameters cannot bias the ratio.
    """

    state_version = 1

    def __init__(
        self,
        *,
        initial_weight: float = 0.1,
        target_ratio: float = 0.1,
        ema_decay: float = 0.95,
        update_every: int = 50,
        min_weight: float = 0.1,
        max_weight: float = 1000.0,
        epsilon: float = 1e-12,
    ) -> None:
        if initial_weight < 0.0:
            raise ValueError("initial_weight must be nonnegative")
        if target_ratio <= 0.0:
            raise ValueError("target_ratio must be positive")
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if update_every < 1:
            raise ValueError("update_every must be positive")
        if min_weight < 0.0 or max_weight < min_weight:
            raise ValueError("OT weight bounds are invalid")
        if not min_weight <= initial_weight <= max_weight:
            raise ValueError("initial_weight must lie within OT weight bounds")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        self.initial_weight = float(initial_weight)
        self.target_ratio = float(target_ratio)
        self.ema_decay = float(ema_decay)
        self.update_every = int(update_every)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.epsilon = float(epsilon)
        self.weight = float(initial_weight)
        self.batch_step = 0
        self.updates = 0
        self.last_base_grad_norm = 0.0
        self.last_ot_grad_norm = 0.0
        self.base_grad_ema: float | None = None
        self.ot_grad_ema: float | None = None

    @staticmethod
    def _shared_gradient_norms(
        base_loss: Tensor,
        ot_loss: Tensor,
        parameters: tuple[nn.Parameter, ...],
    ) -> tuple[float, float, int]:
        base_gradients = torch.autograd.grad(
            base_loss,
            parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        ot_gradients = torch.autograd.grad(
            ot_loss,
            parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        paired = [
            (base_gradient, ot_gradient)
            for base_gradient, ot_gradient in zip(
                base_gradients, ot_gradients, strict=True
            )
            if base_gradient is not None and ot_gradient is not None
        ]
        if not paired:
            raise RuntimeError(
                "Dynamic OT balancing found no parameter shared by the "
                "backbone objective and OT objective"
            )
        base_squared = sum(
            float(gradient.detach().double().square().sum().item())
            for gradient, _ in paired
        )
        ot_squared = sum(
            float(gradient.detach().double().square().sum().item())
            for _, gradient in paired
        )
        base_norm = math.sqrt(base_squared)
        ot_norm = math.sqrt(ot_squared)
        if not math.isfinite(base_norm) or not math.isfinite(ot_norm):
            raise FloatingPointError(
                "Non-finite shared gradient norm in dynamic OT controller"
            )
        return base_norm, ot_norm, len(paired)

    def step(
        self,
        base_loss: Tensor,
        ot_loss: Tensor,
        parameters: Iterable[nn.Parameter],
    ) -> bool:
        """Possibly update the full-strength OT coefficient.

        The first measurement occurs at ``update_every`` rather than at the
        first minibatch.  This gives zero-initialized graph corrections time
        to establish a meaningful backbone-to-graph gradient path.
        """
        self.batch_step += 1
        if self.batch_step % self.update_every != 0:
            return False
        parameter_tuple = tuple(
            parameter for parameter in parameters if parameter.requires_grad
        )
        if not parameter_tuple:
            raise RuntimeError("Dynamic OT balancing has no trainable parameters")
        base_norm, ot_norm, _ = self._shared_gradient_norms(
            base_loss, ot_loss, parameter_tuple
        )
        self.last_base_grad_norm = base_norm
        self.last_ot_grad_norm = ot_norm
        if self.base_grad_ema is None:
            self.base_grad_ema = base_norm
            self.ot_grad_ema = ot_norm
        else:
            assert self.ot_grad_ema is not None
            keep = self.ema_decay
            update = 1.0 - keep
            self.base_grad_ema = keep * self.base_grad_ema + update * base_norm
            self.ot_grad_ema = keep * self.ot_grad_ema + update * ot_norm
        assert self.ot_grad_ema is not None
        proposed = (
            self.target_ratio
            * self.base_grad_ema
            / max(self.ot_grad_ema, self.epsilon)
        )
        self.weight = min(
            self.max_weight, max(self.min_weight, proposed)
        )
        self.updates += 1
        return True

    def diagnostics(self, *, warmup_scale: float) -> dict[str, float]:
        base_ema = self.base_grad_ema or 0.0
        ot_ema = self.ot_grad_ema or 0.0
        effective_weight = warmup_scale * self.weight
        achieved_ratio = (
            effective_weight * ot_ema / base_ema
            if self.base_grad_ema is not None and base_ema > self.epsilon
            else 0.0
        )
        return {
            "dynamic_ot_weight": self.weight,
            "effective_ot_weight": effective_weight,
            "base_shared_grad_norm": self.last_base_grad_norm,
            "ot_shared_grad_norm": self.last_ot_grad_norm,
            "base_shared_grad_ema": base_ema,
            "ot_shared_grad_ema": ot_ema,
            "ot_to_base_grad_ratio": achieved_ratio,
            "ot_balance_updates": float(self.updates),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "state_version": self.state_version,
            "config": {
                "initial_weight": self.initial_weight,
                "target_ratio": self.target_ratio,
                "ema_decay": self.ema_decay,
                "update_every": self.update_every,
                "min_weight": self.min_weight,
                "max_weight": self.max_weight,
                "epsilon": self.epsilon,
            },
            "weight": self.weight,
            "batch_step": self.batch_step,
            "updates": self.updates,
            "last_base_grad_norm": self.last_base_grad_norm,
            "last_ot_grad_norm": self.last_ot_grad_norm,
            "base_grad_ema": self.base_grad_ema,
            "ot_grad_ema": self.ot_grad_ema,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("state_version", 0)) != self.state_version:
            raise ValueError("Unsupported dynamic OT controller state version")
        stored_config = state.get("config")
        if stored_config != self.state_dict()["config"]:
            raise ValueError(
                "Dynamic OT controller arguments do not match the checkpoint"
            )
        self.weight = float(state["weight"])
        self.batch_step = int(state["batch_step"])
        self.updates = int(state["updates"])
        self.last_base_grad_norm = float(state["last_base_grad_norm"])
        self.last_ot_grad_norm = float(state["last_ot_grad_norm"])
        self.base_grad_ema = (
            None
            if state["base_grad_ema"] is None
            else float(state["base_grad_ema"])
        )
        self.ot_grad_ema = (
            None
            if state["ot_grad_ema"] is None
            else float(state["ot_grad_ema"])
        )
