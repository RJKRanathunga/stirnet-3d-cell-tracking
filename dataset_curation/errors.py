# DATASET_CURATION_CANONICAL_SKIP_V1

class CurationError(RuntimeError):
    pass


class ManifestError(CurationError):
    pass


class ArtifactError(CurationError):
    pass
