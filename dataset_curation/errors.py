
class CurationError(RuntimeError):
    pass

class ManifestError(CurationError):
    pass

class ArtifactError(CurationError):
    pass
