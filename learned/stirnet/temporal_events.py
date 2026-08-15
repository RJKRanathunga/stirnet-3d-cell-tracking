from __future__ import annotations

import torch
from torch import Tensor


TEMPORAL_NODE_EVENT_FEATURE_DIM = 8
TEMPORAL_NODE_EVENT_FEATURE_NAMES = (
    "normalized_time",
    "normalized_length_before",
    "normalized_length_after",
    "is_current",
    "is_interior_start",
    "is_interior_end",
    "is_division",
    "is_boundary",
)

# The current 32-D detection-node contract predates the explicit event tensor.
# Keep the one supported legacy mapping centralized here so model code never
# reaches into undocumented graph columns.
LEGACY_GRAPH_EVENT_COLUMNS = (0, 23, 24, 27, 28, 29, 30, 31)


def event_features_from_legacy_graph_x(
    graph_x: Tensor,
    *,
    temporal_radius: int,
) -> Tensor:
    """Reconstruct the explicit [N,8] event contract from legacy 32-D nodes.

    Columns 23 and 24 in the legacy graph contain unnormalized inclusive track
    lengths. Notebook 17 normalized them by the full temporal-window length;
    the same conversion is applied here. No other graph columns are inferred or
    reinterpreted.
    """

    if graph_x.ndim != 2 or graph_x.shape[1] != 32:
        raise ValueError(
            "legacy event reconstruction requires graph_x with shape [N,32]"
        )
    window = float(2 * max(int(temporal_radius), 0) + 1)
    result = torch.stack(
        [
            graph_x[:, LEGACY_GRAPH_EVENT_COLUMNS[0]],
            graph_x[:, LEGACY_GRAPH_EVENT_COLUMNS[1]] / window,
            graph_x[:, LEGACY_GRAPH_EVENT_COLUMNS[2]] / window,
            *[
                graph_x[:, column]
                for column in LEGACY_GRAPH_EVENT_COLUMNS[3:]
            ],
        ],
        dim=-1,
    )
    return result.to(dtype=torch.float32)
