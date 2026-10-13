"""Plot per-step curves saved by evaluate_ablation.py.

Kept separate from the evaluation (and free of torch) because importing
matplotlib after torch on this Windows/conda setup loads a second OpenMP
runtime (libomp.dll next to torch's libiomp5md.dll) and aborts the process.

    python script/jointgv/viz/plot_ablation_curves.py runs/eval/P_final_test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FLOOR_LABEL = "observer on ground truth"
HIGHER_IS_BETTER = {"ssim"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("result_dir", type=Path, help="evaluate_ablation.py --output")
    args = parser.parse_args()
    result = json.loads((args.result_dir / "metrics.json").read_text(encoding="utf-8"))
    curves = result["curves"]
    labels = [model["label"] for model in result["models"]]
    metrics = [name for name in curves[labels[0]]]
    steps = range(1, result["horizon"] + 1)
    figure, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.4))
    for axis, name in zip(axes if len(metrics) > 1 else [axes], metrics):
        for label in labels + ([FLOOR_LABEL] if name == "joint_px" and FLOOR_LABEL in curves else []):
            style = "--" if label == FLOOR_LABEL else "-"
            axis.plot(steps, curves[label][name], style, label=label)
        better = "higher" if name in HIGHER_IS_BETTER else "lower"
        axis.set_title(f"{name} ({better} is better)")
        axis.set_xlabel("rollout step")
        axis.grid(alpha=0.3)
    (axes[0] if len(metrics) > 1 else axes).legend(fontsize=8)
    figure.tight_layout()
    path = args.result_dir / "curves.png"
    figure.savefig(path, dpi=150)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
