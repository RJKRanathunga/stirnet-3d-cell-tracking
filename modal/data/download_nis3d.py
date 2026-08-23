# modal/download_nis3d.py

from pathlib import Path
import hashlib
import shutil
import urllib.request
import zipfile

import modal


app = modal.App("download-nis3d")

external_volume = modal.Volume.from_name(
    "external",
    create_if_missing=True,
)

ZENODO_URL = (
    "https://zenodo.org/records/11456029/files/NIS3D.zip?download=1"
)

EXPECTED_MD5 = "f229013526645c79d31bc58ba7f12be7"

DEST = Path("/external/NIS3D")

SAMPLES = {
    "Drosophila_1",
    "Drosophila_2",
    "MusMusculus_1",
    "MusMusculus_2",
    "Zebrafish_1",
    "Zebrafish_2",
}


def md5_file(path: Path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.md5()

    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


@app.function(
    volumes={"/external": external_volume},
    timeout=60 * 60 * 3,
    cpu=2,
)
def download():
    zip_path = Path("/tmp/NIS3D.zip")
    extract_root = Path("/tmp/NIS3D_extracted")

    # Do not overwrite an existing copy.
    if DEST.exists():
        print(f"NIS3D already exists at {DEST}")
        print("Nothing changed.")
        return

    # ---------------------------------------------------------
    # Download official compressed release
    # ---------------------------------------------------------

    print("Downloading NIS3D from Zenodo...")

    urllib.request.urlretrieve(
        ZENODO_URL,
        zip_path,
    )

    print(
        f"Downloaded archive: "
        f"{zip_path.stat().st_size / 1024**3:.2f} GiB"
    )

    # ---------------------------------------------------------
    # Integrity check
    # ---------------------------------------------------------

    print("Checking MD5...")

    actual_md5 = md5_file(zip_path)

    print(f"Expected: {EXPECTED_MD5}")
    print(f"Actual:   {actual_md5}")

    if actual_md5.lower() != EXPECTED_MD5.lower():
        raise RuntimeError("NIS3D MD5 verification failed.")

    print("MD5 OK.")

    # ---------------------------------------------------------
    # Extract into ephemeral disk
    # ---------------------------------------------------------

    print("Extracting into /tmp...")

    extract_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_root)

    # ---------------------------------------------------------
    # Find canonical six-volume directory
    # ---------------------------------------------------------

    candidates = []

    for path in extract_root.rglob("*"):
        if not path.is_dir():
            continue

        children = {
            child.name
            for child in path.iterdir()
            if child.is_dir()
        }

        if SAMPLES.issubset(children):
            candidates.append(path)

    if not candidates:
        raise RuntimeError(
            "Could not find canonical NIS3D directory."
        )

    canonical = min(
        candidates,
        key=lambda p: len(p.parts),
    )

    print(f"Canonical data found at: {canonical}")

    # ---------------------------------------------------------
    # Copy ONLY original six datasets
    # ---------------------------------------------------------

    DEST.mkdir(
        parents=True,
        exist_ok=True,
    )

    for sample in sorted(SAMPLES):
        src = canonical / sample
        dst = DEST / sample

        print(f"Copying {sample}...")

        shutil.copytree(
            src,
            dst,
        )

    # Persist Volume changes
    external_volume.commit()

    print()
    print("NIS3D installation complete:")
    print("/external/NIS3D")
    print()

    for sample in sorted(SAMPLES):
        print(f"  /external/NIS3D/{sample}")


@app.local_entrypoint()
def main():
    download.remote()