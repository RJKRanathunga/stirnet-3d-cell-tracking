from pathlib import Path
import zarr

# ------------------------------------------------------------
# Project root
# verify.py is located in:
# cell-tracking/analysis/verify.py
#
# Therefore parents[1] = cell-tracking/
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

DATA_ROOT = (
        PROJECT_ROOT
        / "data-sample"
        / "biohub_5samples_20timepoints"
)

SAMPLE_ID = "44b6_0113de3b"

SAMPLE_ROOT = (
        DATA_ROOT
        / "train"
        / SAMPLE_ID
)

ZARR_PATH = (
        SAMPLE_ROOT
        / f"{SAMPLE_ID}.zarr"
)

GROUND_TRUTH_DIR = (
        SAMPLE_ROOT
        / "ground_truth"
)

NODES_CSV = (
        GROUND_TRUTH_DIR
        / "ground_truth_nodes.csv"
)

EDGES_CSV = (
        GROUND_TRUTH_DIR
        / "ground_truth_edges.csv"
)

# ------------------------------------------------------------
# Print paths
# ------------------------------------------------------------

print("=" * 60)
print("DATASET VERIFICATION")
print("=" * 60)

print("\nProject root:")
print(PROJECT_ROOT)

print("\nSample root:")
print(SAMPLE_ROOT)

print("\nZarr path:")
print(ZARR_PATH)

print("\nGround truth nodes:")
print(NODES_CSV)

print("\nGround truth edges:")
print(EDGES_CSV)

# ------------------------------------------------------------
# Check files/directories
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("PATH CHECKS")
print("=" * 60)

print("Sample exists:", SAMPLE_ROOT.exists())
print("Zarr exists:", ZARR_PATH.exists())
print("Ground truth directory exists:", GROUND_TRUTH_DIR.exists())
print("Nodes CSV exists:", NODES_CSV.exists())
print("Edges CSV exists:", EDGES_CSV.exists())

if not ZARR_PATH.exists():
    raise FileNotFoundError(
        f"\nZarr directory not found:\n{ZARR_PATH}"
    )

# ------------------------------------------------------------
# Open Zarr v3 array
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("OPENING ZARR")
print("=" * 60)

ARRAY_PATH = ZARR_PATH / "0"

print("Array path:")
print(ARRAY_PATH)

print("Array metadata exists:")
print((ARRAY_PATH / "zarr.json").exists())

image = zarr.open_array(
    str(ARRAY_PATH),
    mode="r"
)

print("\nSUCCESSFULLY OPENED ARRAY")

print("Array:")
print(image)

print("Shape:")
print(image.shape)

print("Dtype:")
print(image.dtype)

print("Chunks:")
print(image.chunks)

# ------------------------------------------------------------
# Read first timepoint
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FIRST TIMEPOINT")
print("=" * 60)

frame = image[0]

print("Frame shape:", frame.shape)
print("Frame dtype:", frame.dtype)
print("Minimum:", frame.min())
print("Maximum:", frame.max())
print("Mean:", frame.mean())