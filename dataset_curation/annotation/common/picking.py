
from importlib import import_module

def instance_helper(name: str):
    return getattr(import_module("dataset_curation._compat.instance_annotator"), name)

def track_helper(name: str):
    return getattr(import_module("dataset_curation._compat.track_annotator"), name)
