"""Deterministic checks for the RM65-B cart-pole rollout metrics.

Run from the repository root:
    python script/jointgv/tests/check_rollout_metrics.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jointgv.rollout_metrics import cartpole_structure_scores


def main() -> None:
    roles = ("base", "link_1", "link_2", "link_3", "end_effector", "object_1")
    target = torch.zeros(1, 2, len(roles), 2)
    predicted = torch.zeros_like(target)
    visible = torch.ones(1, 2, len(roles))

    # Step 1: both poles are upright; the prediction is translated 0.02 in x.
    target[0, 0, 4] = torch.tensor([0.10, 0.0])
    target[0, 0, 5] = torch.tensor([0.10, -1.0])
    predicted[0, 0, 4] = torch.tensor([0.12, 0.0])
    predicted[0, 0, 5] = torch.tensor([0.12, -1.0])

    # Step 2: ground truth has fallen 90 degrees; prediction stays upright.
    target[0, 1, 4] = torch.tensor([0.20, 0.0])
    target[0, 1, 5] = torch.tensor([1.20, 0.0])
    predicted[0, 1, 4] = torch.tensor([0.20, 0.0])
    predicted[0, 1, 5] = torch.tensor([0.20, -1.0])

    first_target = torch.zeros(1, len(roles), 2)
    first_predicted = torch.zeros_like(first_target)
    first_visible = torch.ones(1, len(roles))
    first_target[0, 5] = torch.tensor([0.0, -1.0])
    first_predicted[0, 5] = torch.tensor([0.0, -1.0])

    scores = cartpole_structure_scores(
        predicted,
        target,
        visible,
        role_names=roles,
        image_size=101,
        first_predicted_positions=first_predicted,
        first_target_positions=first_target,
        first_visible=first_visible,
    )
    torch.testing.assert_close(scores["hinge_px"], torch.tensor([[1.0, 0.0]]))
    torch.testing.assert_close(
        scores["pole_tip_px"], torch.tensor([[1.0, 2.0**0.5 * 50.0]])
    )
    torch.testing.assert_close(
        scores["pole_vector_px"], torch.tensor([[0.0, 2.0**0.5 * 50.0]])
    )
    torch.testing.assert_close(scores["pole_angle_deg"], torch.tensor([[0.0, 90.0]]))
    torch.testing.assert_close(
        scores["pole_angle_change_deg"], torch.tensor([[0.0, 90.0]])
    )

    # A hidden tip invalidates every metric that depends on the pole vector,
    # including the change into that step, while hinge localization remains.
    hidden = visible.clone()
    hidden[0, 1, 5] = 0.0
    hidden_scores = cartpole_structure_scores(
        predicted,
        target,
        hidden,
        role_names=roles,
        image_size=101,
        first_predicted_positions=first_predicted,
        first_target_positions=first_target,
        first_visible=first_visible,
    )
    assert torch.isnan(hidden_scores["pole_tip_px"][0, 1])
    assert torch.isnan(hidden_scores["pole_vector_px"][0, 1])
    assert torch.isnan(hidden_scores["pole_angle_deg"][0, 1])
    assert torch.isnan(hidden_scores["pole_angle_change_deg"][0, 1])
    assert torch.isfinite(hidden_scores["hinge_px"][0, 1])
    print("cart-pole rollout metric checks passed")


if __name__ == "__main__":
    main()
