
from __future__ import annotations
from dataset_curation.workspace.sample import CurationSample

def register_source(**kwargs) -> CurationSample:
    return CurationSample.create(**kwargs)
