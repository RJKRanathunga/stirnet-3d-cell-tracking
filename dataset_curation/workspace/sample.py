
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path

from dataset_curation._repo import repo_root
from dataset_curation.config import DEFAULT_DATASET, DEFAULT_SPACING_ZYX_UM, curation_root
from dataset_curation.errors import ManifestError
from dataset_curation.io.atomic import atomic_json, read_json
from .layout import CurationLayout

SCHEMA_VERSION = 1

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

class CurationSample:
    def __init__(self, layout: CurationLayout):
        self.layout = layout

    @classmethod
    def create(
        cls,
        *,
        sample_id: str,
        source_zarr: str | Path,
        root: str | Path | None = None,
        dataset: str = DEFAULT_DATASET,
        spacing_zyx_um=DEFAULT_SPACING_ZYX_UM,
    ):
        layout = CurationLayout(curation_root(root), dataset, sample_id)
        layout.sample_root.mkdir(parents=True, exist_ok=True)
        layout.prepared.mkdir(parents=True, exist_ok=True)
        layout.inference.mkdir(parents=True, exist_ok=True)
        layout.annotations.mkdir(parents=True, exist_ok=True)

        source = Path(source_zarr).expanduser()
        source = source.resolve() if source.is_absolute() else (repo_root() / source).resolve()
        if not source.exists():
            raise FileNotFoundError(source)

        old = read_json(layout.sample_manifest) if layout.sample_manifest.is_file() else {}
        atomic_json(
            layout.sample_manifest,
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "curation_sample",
                "dataset": dataset,
                "sample_id": sample_id,
                "source_zarr": str(source),
                "spacing_zyx_um": [float(v) for v in spacing_zyx_um],
                "created_at": old.get("created_at", _now()),
                "updated_at": _now(),
            },
        )
        return cls(layout)

    @classmethod
    def open(cls, *, sample_id: str, root=None, dataset: str = DEFAULT_DATASET):
        layout = CurationLayout(curation_root(root), dataset, sample_id)
        if not layout.sample_manifest.is_file():
            raise FileNotFoundError(
                f"Sample is not registered: {layout.sample_manifest}. Run setup first."
            )
        payload = read_json(layout.sample_manifest)
        if payload.get("sample_id") != sample_id:
            raise ManifestError("Sample manifest ID mismatch.")
        return cls(layout)

    @property
    def manifest(self) -> dict:
        return read_json(self.layout.sample_manifest)

    @property
    def source_zarr(self) -> Path:
        return Path(self.manifest["source_zarr"])

    @property
    def spacing_zyx_um(self):
        return tuple(float(v) for v in self.manifest["spacing_zyx_um"])

    def ensure_inference_run(self, run_id: str) -> Path:
        root = self.layout.inference_run(run_id)
        root.mkdir(parents=True, exist_ok=True)
        return root

    def ensure_annotation_set(self, name: str, *, base_inference_run: str) -> Path:
        root = self.layout.annotation_set(name)
        root.mkdir(parents=True, exist_ok=True)
        self.layout.instance_annotations(name).mkdir(parents=True, exist_ok=True)
        self.layout.track_annotations(name).mkdir(parents=True, exist_ok=True)
        self.layout.point_annotations(name).mkdir(parents=True, exist_ok=True)

        path = self.layout.annotation_manifest(name)
        if path.is_file():
            payload = read_json(path)
            existing = payload.get("base_inference_run")
            if existing != base_inference_run:
                raise ManifestError(
                    f"Annotation set {name!r} is bound to {existing!r}, "
                    f"not {base_inference_run!r}."
                )
        else:
            atomic_json(
                path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": "annotation_set",
                    "sample_id": self.layout.sample_id,
                    "annotation_set": name,
                    "base_inference_run": base_inference_run,
                    "created_at": _now(),
                },
            )
        return root
