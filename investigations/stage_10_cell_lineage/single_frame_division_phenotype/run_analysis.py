"""CLI for the single-frame division phenotype investigation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import PhenotypeInvestigationConfig
from .pipeline import run_investigation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare manually labelled parent/daughter cells with all other cells in the same frame."
    )
    parser.add_argument("--scenes-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--local-radius-um", type=float, default=25.0)
    parser.add_argument("--boundary-margin-um", type=float, default=4.0)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-galleries", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = PhenotypeInvestigationConfig(
        scenes_root=args.scenes_root,
        local_population_radius_um=args.local_radius_um,
        boundary_margin_um=args.boundary_margin_um,
        create_plots=not args.no_plots,
        create_galleries=not args.no_galleries,
        strict=args.strict,
    )
    result = run_investigation(config, output_directory=args.output_dir, progress=print)
    print(f"Output: {result.output_directory}")
    print(f"Valid scenes: {result.metadata['valid_scene_count']}")
    print(f"Target observations: {result.metadata['target_observation_count']}")
    print(f"Full-population cell observations: {result.metadata['full_population_cell_observation_count']}")
    print("Top parent features:")
    parent = result.feature_effect_summary.loc[
        result.feature_effect_summary["phenotype_subtype"].eq("parent_final")
        & result.feature_effect_summary["population"].eq("all_other_cells")
    ].head(15)
    print(parent[["feature", "median_percentile", "extreme_direction_consistency"]].to_string(index=False))
    print("Top daughter features:")
    daughter = result.feature_effect_summary.loc[
        result.feature_effect_summary["phenotype_subtype"].eq("daughter_birth")
        & result.feature_effect_summary["population"].eq("all_other_cells")
    ].head(15)
    print(daughter[["feature", "median_percentile", "extreme_direction_consistency"]].to_string(index=False))


if __name__ == "__main__":
    main()
