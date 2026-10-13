"""Motion- and structure-aware rollout metrics shared by training and eval.

Pixel MSE favours blur: a sharp limb drawn slightly off is penalized twice
(missing where it should be, present where it should not), a smeared one
less. These metrics measure what the long-horizon claim is about instead:

* motion-weighted MSE: per-pixel error weighted by the ground truth's own
  frame-to-frame change (evaluate_rollout.py's definition);
* SSIM (Gaussian 11x11 window, sigma 1.5);
* joint-position error: one fixed, independently pretrained Explicit Graph
  Observer reads the predicted frames and its node 2D positions are compared
  with the ground-truth graph_node_pixel labels over visible nodes, in
  pixels. The same observer is used for every model (the graph-off baseline
  has none of its own, and graph-on runs fine-tune theirs).
* robot cart-pole structure errors: hinge and pole-tip pixel errors, pole-vector
  pixel error, wrapped pole-angle MAE, and wrapped pole angular-change error.
  These are emitted only when the observer schema contains ``end_effector``
  (the passive hinge) and ``object_1`` (the pole tip).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
import torch.nn.functional as F

from .blocks.explicit_graph_observer import ExplicitGraphObserver


def load_reference_observer(path: Path, device: torch.device) -> ExplicitGraphObserver:
    """The pretrain_graph_observer.py checkpoint, frozen, as a measuring tool."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = dict(payload["observer_config"])
    # The checkpoint already holds the RelTR weights.
    config["pretrained_backbone"] = False
    config["reltr_batch_size"] = max(int(config.get("reltr_batch_size", 8)), 64)
    observer = ExplicitGraphObserver(**config)
    observer.load_state_dict(payload["model"])
    observer.set_temporal_strength(float(payload.get("temporal_strength", 1.0)))
    observer.requires_grad_(False)
    return observer.to(device).eval()


def gaussian_window(device: torch.device, size: int = 11, sigma: float = 1.5) -> Tensor:
    coords = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return (kernel[:, None] * kernel[None, :]).expand(3, 1, size, size).contiguous()


def ssim(prediction: Tensor, target: Tensor, window: Tensor) -> Tensor:
    """Per-image SSIM of [N,3,H,W] images in [0,1]."""
    c1, c2 = 0.01**2, 0.03**2

    def blur(value: Tensor) -> Tensor:
        return F.conv2d(value, window, groups=3)

    mu_p, mu_t = blur(prediction), blur(target)
    var_p = blur(prediction * prediction) - mu_p**2
    var_t = blur(target * target) - mu_t**2
    covariance = blur(prediction * target) - mu_p * mu_t
    value = ((2 * mu_p * mu_t + c1) * (2 * covariance + c2)) / (
        (mu_p**2 + mu_t**2 + c1) * (var_p + var_t + c2)
    )
    return value.mean(dim=(1, 2, 3))


def motion_weighted_mse(predicted: Tensor, target: Tensor, previous: Tensor) -> Tensor:
    """[B,H] error of [B,H,3,h,w] frames, weighted by ground-truth motion."""
    motion = (target - previous).abs().mul(255.0).mean(dim=2) + 1.0
    motion = motion / motion.mean(dim=(-2, -1), keepdim=True)
    return ((predicted - target).square().mean(dim=2) * motion).mean(dim=(-2, -1))


def joint_error_px(
    observer: ExplicitGraphObserver,
    first: Tensor,
    frames: Tensor,
    pixel: Tensor,
    visible: Tensor,
) -> Tensor:
    """Observer node positions on ``frames`` vs labels, [B,H] pixels.

    The real start frame is prepended so the observer's temporal recurrence
    starts from the same real frame for predictions and ground truth; its
    own output is dropped. Windows with no visible node give NaN.
    """
    sequence = torch.cat((first[:, None], frames), dim=1)
    positions = observer.forward_sequence(sequence)["positions_2d"][:, 1:]
    return _joint_error_from_positions(positions, pixel, visible, frames.shape[-1])


def _masked(values: Tensor, valid: Tensor) -> Tensor:
    return torch.where(valid, values, torch.full_like(values, float("nan")))


def _joint_error_from_positions(
    positions: Tensor,
    pixel: Tensor,
    visible: Tensor,
    image_size: int,
) -> Tensor:
    distance = (positions - pixel).norm(dim=-1) * (image_size - 1) / 2.0
    count = visible.sum(dim=-1)
    error = (distance * visible).sum(dim=-1) / count.clamp_min(1.0)
    return _masked(error, count > 0)


def _wrapped_angle_difference(first: Tensor, second: Tensor) -> Tensor:
    """Signed, wrapped ``first - second`` in radians."""
    difference = first - second
    return torch.atan2(torch.sin(difference), torch.cos(difference))


def cartpole_structure_scores(
    predicted_positions: Tensor,
    target_positions: Tensor,
    visible: Tensor,
    *,
    role_names: tuple[str, ...] | list[str],
    image_size: int,
    first_predicted_positions: Tensor | None = None,
    first_target_positions: Tensor | None = None,
    first_visible: Tensor | None = None,
) -> dict[str, Tensor]:
    """Cart-pole errors per rollout step, each ``[B,H]``.

    Positions use the dataset's normalized image coordinates.  The
    ``end_effector`` role is the passive pole hinge and ``object_1`` is the
    pole tip.  Angular-change error needs the real start-frame labels and is
    omitted when those optional tensors are unavailable.
    """
    names = tuple(role_names)
    if "end_effector" not in names or "object_1" not in names:
        return {}
    hinge_index = names.index("end_effector")
    tip_index = names.index("object_1")
    scale = (image_size - 1) / 2.0

    predicted_hinge = predicted_positions[..., hinge_index, :]
    predicted_tip = predicted_positions[..., tip_index, :]
    target_hinge = target_positions[..., hinge_index, :]
    target_tip = target_positions[..., tip_index, :]
    hinge_visible = visible[..., hinge_index] > 0.5
    tip_visible = visible[..., tip_index] > 0.5
    pole_visible = hinge_visible & tip_visible

    hinge_error = (predicted_hinge - target_hinge).norm(dim=-1) * scale
    tip_error = (predicted_tip - target_tip).norm(dim=-1) * scale
    predicted_vector = predicted_tip - predicted_hinge
    target_vector = target_tip - target_hinge
    vector_error = (predicted_vector - target_vector).norm(dim=-1) * scale
    predicted_norm = predicted_vector.norm(dim=-1)
    target_norm = target_vector.norm(dim=-1)
    angle_valid = pole_visible & (predicted_norm > 1e-6) & (target_norm > 1e-6)
    predicted_angle = torch.atan2(predicted_vector[..., 1], predicted_vector[..., 0])
    target_angle = torch.atan2(target_vector[..., 1], target_vector[..., 0])
    angle_error = _wrapped_angle_difference(predicted_angle, target_angle).abs()

    result = {
        "pole_tip_px": _masked(tip_error, tip_visible),
        "hinge_px": _masked(hinge_error, hinge_visible),
        "pole_vector_px": _masked(vector_error, pole_visible),
        "pole_angle_deg": _masked(angle_error.rad2deg(), angle_valid),
    }
    if (
        first_predicted_positions is None
        or first_target_positions is None
        or first_visible is None
    ):
        return result

    first_predicted_hinge = first_predicted_positions[..., hinge_index, :]
    first_predicted_tip = first_predicted_positions[..., tip_index, :]
    first_target_hinge = first_target_positions[..., hinge_index, :]
    first_target_tip = first_target_positions[..., tip_index, :]
    first_predicted_vector = first_predicted_tip - first_predicted_hinge
    first_target_vector = first_target_tip - first_target_hinge
    first_pole_visible = (
        (first_visible[..., hinge_index] > 0.5)
        & (first_visible[..., tip_index] > 0.5)
    )
    first_angle_valid = (
        first_pole_visible
        & (first_predicted_vector.norm(dim=-1) > 1e-6)
        & (first_target_vector.norm(dim=-1) > 1e-6)
    )
    first_predicted_angle = torch.atan2(
        first_predicted_vector[..., 1], first_predicted_vector[..., 0]
    )
    first_target_angle = torch.atan2(
        first_target_vector[..., 1], first_target_vector[..., 0]
    )
    predicted_angles = torch.cat((first_predicted_angle[:, None], predicted_angle), dim=1)
    target_angles = torch.cat((first_target_angle[:, None], target_angle), dim=1)
    predicted_change = _wrapped_angle_difference(
        predicted_angles[:, 1:], predicted_angles[:, :-1]
    )
    target_change = _wrapped_angle_difference(
        target_angles[:, 1:], target_angles[:, :-1]
    )
    change_error = _wrapped_angle_difference(predicted_change, target_change).abs()
    all_valid = torch.cat((first_angle_valid[:, None], angle_valid), dim=1)
    change_valid = all_valid[:, 1:] & all_valid[:, :-1]
    result["pole_angle_change_deg"] = _masked(change_error.rad2deg(), change_valid)
    return result


def rollout_scores(
    predicted: Tensor,
    first: Tensor,
    target: Tensor,
    previous: Tensor,
    pixel: Tensor,
    visible: Tensor,
    observer: ExplicitGraphObserver,
    window: Tensor,
    *,
    first_pixel: Tensor | None = None,
    first_visible: Tensor | None = None,
) -> dict[str, Tensor]:
    """Motion, image and available structure metrics per step, each [B,H]."""
    batch, horizon = target.shape[:2]
    predicted = predicted.clamp(0.0, 1.0)
    sequence = torch.cat((first[:, None], predicted), dim=1)
    all_positions = observer.forward_sequence(sequence)["positions_2d"]
    predicted_positions = all_positions[:, 1:]
    result = {
        "motion_mse": motion_weighted_mse(predicted, target, previous),
        "ssim": ssim(predicted.flatten(0, 1), target.flatten(0, 1), window).view(
            batch, horizon
        ),
        "joint_px": _joint_error_from_positions(
            predicted_positions, pixel, visible, predicted.shape[-1]
        ),
    }
    result.update(
        cartpole_structure_scores(
            predicted_positions,
            pixel,
            visible,
            role_names=observer.role_names,
            image_size=predicted.shape[-1],
            first_predicted_positions=all_positions[:, 0],
            first_target_positions=first_pixel,
            first_visible=first_visible,
        )
    )
    return result
