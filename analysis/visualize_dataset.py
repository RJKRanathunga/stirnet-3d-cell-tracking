from pathlib import Path

import numpy as np
import pandas as pd
import zarr
import napari


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = (
        PROJECT_ROOT
        / "data-sample"
        / "biohub_5samples_20timepoints"
        / "train"
)

# Physical voxel size in micrometers
# Z is much coarser than X/Y
VOXEL_SIZE = (
    1.625,      # Z
    0.40625,    # Y
    0.40625,    # X
)


# ============================================================
# FIND DATASETS
# ============================================================

def find_datasets():
    """
    Find all extracted samples.

    Expected structure:

    train/
        sample_id/
            sample_id.zarr/
                0/
                    zarr.json

            ground_truth/
                ground_truth_nodes.csv
                ground_truth_edges.csv
    """

    datasets = []

    for sample_dir in sorted(DATA_ROOT.iterdir()):

        if not sample_dir.is_dir():
            continue

        zarr_path = sample_dir / f"{sample_dir.name}.zarr"

        gt_dir = sample_dir / "ground_truth"

        nodes_path = gt_dir / "ground_truth_nodes.csv"

        edges_path = gt_dir / "ground_truth_edges.csv"

        if (
                zarr_path.exists()
                and nodes_path.exists()
                and edges_path.exists()
        ):
            datasets.append(
                {
                    "name": sample_dir.name,
                    "root": sample_dir,
                    "zarr": zarr_path,
                    "nodes": nodes_path,
                    "edges": edges_path,
                }
            )

    return datasets


# ============================================================
# OPEN ZARR
# ============================================================

def load_zarr(zarr_path):
    """
    Open the actual array at path '0'.

    The Zarr root is a Group.
    The image array is stored at:

        sample.zarr/
            0/
                zarr.json
                c/
    """

    print()
    print("=" * 70)
    print("OPENING ZARR")
    print("=" * 70)

    print("Path:")
    print(zarr_path)

    # Open array directly at path "0"
    array = zarr.open_array(
        str(zarr_path),
        mode="r",
        path="0",
    )

    print()
    print("Array:")
    print(array)

    print("Shape:")
    print(array.shape)

    print("Dtype:")
    print(array.dtype)

    print("Chunks:")
    print(array.chunks)

    return array


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth(nodes_path, edges_path):

    print()
    print("=" * 70)
    print("LOADING GROUND TRUTH")
    print("=" * 70)

    nodes = pd.read_csv(nodes_path)

    edges = pd.read_csv(edges_path)

    print()
    print("Nodes:")
    print(nodes.head())

    print()
    print("Edges:")
    print(edges.head())

    print()
    print("Number of nodes:", len(nodes))
    print("Number of edges:", len(edges))

    print()
    print("Node columns:")
    print(nodes.columns.tolist())

    print()
    print("Edge columns:")
    print(edges.columns.tolist())

    return nodes, edges


# ============================================================
# PREPARE GROUND TRUTH POINTS
# ============================================================

def prepare_points(nodes):

    required_columns = [
        "node_id",
        "t",
        "z",
        "y",
        "x",
    ]

    for column in required_columns:

        if column not in nodes.columns:

            raise ValueError(
                f"Missing required column: {column}"
            )

    # napari expects:
    #
    # [T, Z, Y, X]
    #
    # for a 4D points layer

    points = nodes[
        ["t", "z", "y", "x"]
    ].to_numpy(
        dtype=np.float32
    )

    return points


# ============================================================
# MAIN VISUALIZATION
# ============================================================

def visualize_dataset(dataset):

    name = dataset["name"]

    print()
    print()
    print("#" * 70)
    print(f"VISUALIZING DATASET: {name}")
    print("#" * 70)

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = load_zarr(
        dataset["zarr"]
    )

    # --------------------------------------------------------
    # Load ground truth
    # --------------------------------------------------------

    nodes, edges = load_ground_truth(
        dataset["nodes"],
        dataset["edges"],
    )

    # --------------------------------------------------------
    # Prepare points
    # --------------------------------------------------------

    points = prepare_points(
        nodes
    )

    print()
    print("Ground-truth point array:")
    print(points.shape)

    # --------------------------------------------------------
    # Create napari viewer
    # --------------------------------------------------------

    viewer = napari.Viewer(
        title=f"BioHub Cell Tracking - {name}",
        ndisplay=2
    )

    # --------------------------------------------------------
    # Add image
    # --------------------------------------------------------

    viewer.add_image(
        image,
        name=f"{name} - Fluorescence",
        scale=(1, *VOXEL_SIZE),
        contrast_limits=(
            0,
            3000,
        ),
        colormap="gray",
        rendering="attenuated_mip",
        opacity=0.7,
    )

    # --------------------------------------------------------
    # Add ground truth points
    # --------------------------------------------------------

    viewer.add_points(
        points,
        name="Ground Truth Cells",
        scale=(1, *VOXEL_SIZE),
        size=2.5,
        face_color="red",
        ndim=4,
    )

    # --------------------------------------------------------
    # Add text information
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("NAPARI CONTROLS")
    print("=" * 70)

    print()
    print("Dimensions:")
    print("  T = Time")
    print("  Z = Depth")
    print("  Y = Height")
    print("  X = Width")

    print()
    print("Use the dimension sliders to:")
    print("  - Move through time")
    print("  - Move through Z slices")

    print()
    print("3D visualization:")
    print("  - Click the 3D button in napari")
    print("  - Rotate the volume")
    print("  - Zoom")
    print("  - Inspect cell positions")

    print()
    print("Dataset:")
    print(name)

    print()
    print("Image shape:")
    print(image.shape)

    print()
    print("Ground truth nodes:")
    print(len(nodes))

    print()
    print("Ground truth edges:")
    print(len(edges))

    print()

    # --------------------------------------------------------
    # Start napari
    # --------------------------------------------------------

    napari.run()


# ============================================================
# DATASET SELECTION
# ============================================================

def main():

    print("=" * 70)
    print("BIOHUB CELL TRACKING VISUALIZER")
    print("=" * 70)

    print()
    print("Data root:")
    print(DATA_ROOT)

    if not DATA_ROOT.exists():

        raise FileNotFoundError(
            f"Data directory not found:\n{DATA_ROOT}"
        )

    datasets = find_datasets()

    if len(datasets) == 0:

        raise RuntimeError(
            "No valid datasets found."
        )

    print()
    print(f"Found {len(datasets)} dataset(s):")

    for i, dataset in enumerate(
            datasets,
            start=1,
    ):

        print(
            f"  [{i}] {dataset['name']}"
        )

    print()

    # --------------------------------------------------------
    # Select dataset
    # --------------------------------------------------------

    while True:

        choice = input(
            "Select dataset number: "
        ).strip()

        try:

            index = int(choice) - 1

            if 0 <= index < len(datasets):

                break

        except ValueError:

            pass

        print(
            "Invalid selection. Try again."
        )

    dataset = datasets[index]

    # --------------------------------------------------------
    # Visualize
    # --------------------------------------------------------

    visualize_dataset(
        dataset
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()