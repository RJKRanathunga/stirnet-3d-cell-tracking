
import runpy

def run_legacy() -> None:
    runpy.run_module("dataset_curation._compat.point_annotator", run_name="__main__")
