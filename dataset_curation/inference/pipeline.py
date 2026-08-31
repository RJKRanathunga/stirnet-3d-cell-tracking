
from dataset_curation.inference.backends.investigation36 import Investigation36Backend

def run_current_inference(sample, *, run_id: str, extra_args=()):
    return Investigation36Backend().run(sample, run_id=run_id, extra_args=tuple(extra_args))
