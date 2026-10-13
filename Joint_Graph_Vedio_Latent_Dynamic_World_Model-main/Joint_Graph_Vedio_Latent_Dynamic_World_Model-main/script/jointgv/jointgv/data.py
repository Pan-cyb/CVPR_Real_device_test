from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class EpisodeSequences(Dataset):
    """Fixed-length windows of T actions and T+1 images from episode files."""

    def __init__(
        self,
        root: str | Path,
        sequence_length: int = 16,
        stride: int | None = None,
        include_graph_targets: bool | None = None,
        include_control_targets: bool = False,
        episode_files: Sequence[str | Path] | None = None,
    ):
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        self.files = (
            sorted(Path(path) for path in episode_files)
            if episode_files is not None
            else sorted(Path(root).glob("*.npz"))
        )
        if not self.files:
            raise FileNotFoundError(f"No .npz episodes found in {root}")
        self.sequence_length = sequence_length
        self.stride = stride or sequence_length
        self.index: list[tuple[int, int]] = []
        self.lengths = []
        graph_keys = {"graph_node_state", "graph_node_mask", "graph_relation"}
        graph_availability = []
        graph_pixel_availability = []
        for file_id, path in enumerate(self.files):
            with np.load(path) as episode:
                length = min(len(episode["image"]) - 1, len(episode["action"]))
                graph_availability.append(graph_keys.issubset(episode.files))
                graph_pixel_availability.append(
                    {"graph_node_pixel", "graph_node_visible"}.issubset(
                        episode.files
                    )
                )
            self.lengths.append(length)
            last_start = length - sequence_length
            self.index.extend(
                (file_id, start)
                for start in range(0, last_start + 1, self.stride)
            )
        if not self.index:
            longest = max(self.lengths)
            raise ValueError(
                f"No episode contains {sequence_length} transitions; "
                f"longest episode has {longest}"
            )
        self.has_graph_targets = (
            all(graph_availability)
            if include_graph_targets is None
            else include_graph_targets
        )
        if self.has_graph_targets and not all(graph_availability):
            raise ValueError(
                "Graph supervision requested, but some episodes do not contain "
                "graph_node_state, graph_node_mask, and graph_relation"
            )
        self.has_graph_pixels = self.has_graph_targets and all(
            graph_pixel_availability
        )
        self.include_control_targets = include_control_targets
        if include_control_targets:
            for path in self.files:
                with np.load(path) as episode:
                    if "reward" not in episode:
                        raise ValueError(
                            f"Control targets requested, but {path} has no reward"
                        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(
        self, item: int
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor, torch.Tensor, dict[str, torch.Tensor]
    ]:
        file_id, start = self.index[item]
        stop = start + self.sequence_length
        with np.load(self.files[file_id]) as episode:
            images = episode["image"][start:stop + 1]
            actions = episode["action"][start:stop]
            targets: dict[str, torch.Tensor] = {}
            if self.include_control_targets:
                targets["reward"] = torch.from_numpy(
                    episode["reward"][start:stop].copy()
                ).float()
                # Burchim's DreamerV3 discount head targets ``1 - done`` and
                # its DMC wrapper defines done as ``timestep.last()``.  This
                # differs from the environment discount at time limits.
                if "is_last" in episode:
                    continuation = 1.0 - episode["is_last"][
                        start:stop
                    ].astype(np.float32, copy=True)
                elif "continuation" in episode:
                    continuation = episode["continuation"][start:stop].copy()
                elif "discount" in episode:
                    continuation = episode["discount"][start:stop].copy()
                else:
                    continuation = np.ones(stop - start, dtype=np.float32)
                    if stop >= self.lengths[file_id]:
                        continuation[-1] = 0.0
                targets["continuation"] = torch.from_numpy(
                    continuation
                ).float()
            if self.has_graph_targets:
                targets.update(
                    {
                        "node_state": torch.from_numpy(
                            episode["graph_node_state"][
                                start : stop + 1
                            ].copy()
                        ).float(),
                        "node_mask": torch.from_numpy(
                            episode["graph_node_mask"][
                                start : stop + 1
                            ].copy()
                        ).float(),
                        "relation": torch.from_numpy(
                            episode["graph_relation"][
                                start : stop + 1
                            ].copy()
                        ).long(),
                    }
                )
                if self.has_graph_pixels:
                    targets["node_pixel"] = torch.from_numpy(
                        episode["graph_node_pixel"][start : stop + 1].copy()
                    ).float()
                    targets["node_visible"] = torch.from_numpy(
                        episode["graph_node_visible"][start : stop + 1].copy()
                    ).float()
        images = torch.from_numpy(images.copy()).permute(0, 3, 1, 2).float() / 255.0
        action_tensor = torch.from_numpy(actions.copy()).float()
        if not targets:
            return images, action_tensor
        return images, action_tensor, targets


class EpisodePairs(EpisodeSequences):
    """Compatibility alias for one-step datasets."""

    def __init__(self, root: str | Path):
        super().__init__(
            root, sequence_length=1, stride=1, include_graph_targets=False
        )

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images, actions = super().__getitem__(item)
        return images[0], actions[0], images[1]
