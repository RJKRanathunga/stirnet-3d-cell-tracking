"""Verbatim Stage 7 configuration migrated from the reference notebook."""

import numpy as np

VOLUME_SHAPE_ZYX = np.asarray([64, 256, 256], dtype=int)
VOXEL_SIZE_ZYX = np.asarray([1.625, 0.40625, 0.40625], dtype=float)

# ============================================================
# Probabilistic global-relative tracking configuration
# ============================================================

# ------------------------------------------------------------
# Robust global-shift estimation
# ------------------------------------------------------------

GLOBAL_SHIFT_MAX_PAIR_DISTANCE_UM = 12.0
GLOBAL_SHIFT_MAD_SCALE = 3.5
GLOBAL_SHIFT_MIN_INLIER_RADIUS_UM = 1.0
GLOBAL_SHIFT_CONFIDENCE_PAIR_COUNT = 40
GLOBAL_SHIFT_CONFIDENCE_DISPERSION_UM = 2.0

# Second-pass global-shift refinement now uses association
# probability rather than the previous normalized-cost threshold.
GLOBAL_REFINEMENT_MIN_MATCHES = 20
GLOBAL_REFINEMENT_MIN_ASSOCIATION_PROBABILITY = 0.20
GLOBAL_REFINEMENT_MIN_PROBABILITY_MARGIN = 0.02
GLOBAL_REFINEMENT_MAX_PREDICTION_ERROR_UM = 3.5
GLOBAL_REFINEMENT_OBJECTIVE_TOLERANCE = 0.02
GLOBAL_REFINEMENT_MAX_MATCH_LOSS = 2

# ------------------------------------------------------------
# Global-relative motion state
# ------------------------------------------------------------

RELATIVE_VELOCITY_EMA_ALPHA = 0.50
RELATIVE_ERROR_EMA_ALPHA = 0.25
RELATIVE_FULL_CONFIDENCE_SAMPLES = 4
RELATIVE_ERROR_CONFIDENCE_SCALE_UM = 2.0
RELATIVE_MOTION_GAP_DECAY = 0.65
BOUNDARY_RELATIVE_MOTION_CONFIDENCE_SCALE = 0.35
RELATIVE_MOTION_COST_SCALE_UM = 3.0

# Only confident immediate-frame matches may update relative
# velocity and per-track prediction uncertainty.
RELATIVE_UPDATE_MIN_GLOBAL_CONFIDENCE = 0.20
RELATIVE_UPDATE_MIN_ASSOCIATION_PROBABILITY = 0.20
RELATIVE_UPDATE_MIN_PROBABILITY_MARGIN = 0.02
RELATIVE_UPDATE_MAX_DISTANCE_UM = 5.5
RELATIVE_UPDATE_MAX_PAIR_COST = 6.0
MAX_RELATIVE_VELOCITY_UM_PER_FRAME = 5.0

POSITION_RESIDUAL_EMA_ALPHA = 0.20
POSITION_RESIDUAL_UPDATE_MIN_ASSOCIATION_PROBABILITY = 0.20
POSITION_RESIDUAL_UPDATE_MIN_MARGIN = 0.02

# ------------------------------------------------------------
# Adaptive position likelihood
# ------------------------------------------------------------

# Physical uncertainty is axis-specific because centroid estimates
# are less precise along the coarser Z direction.
BASE_POSITION_SIGMA_ZYX_UM = np.asarray(
    [2.40, 1.60, 1.60],
    dtype=float,
)

POSITION_SIGMA_GLOBAL_UNCERTAINTY_UM = 1.50
POSITION_SIGMA_PER_MISSING_FRAME_UM = 1.25
POSITION_SIGMA_BOUNDARY_SCALE = 1.50
POSITION_SIGMA_TRACK_RESIDUAL_WEIGHT = 0.75
POSITION_STUDENT_T_DOF = 4.0

# ------------------------------------------------------------
# Smooth volume and appearance likelihoods
# ------------------------------------------------------------

# Log-ratio scales. A scale of log(1.35), for example, treats an
# approximately 35% change as meaningful but not impossible.
VOLUME_LOG_SCALE_INTERIOR = float(np.log(1.25))
VOLUME_LOG_SCALE_BOUNDARY = float(np.log(2.50))
VOLUME_STUDENT_T_DOF = 8.0

SIZE_RELATIVE_SCALE = 0.30
SHAPE_RELATIVE_SCALE = 0.25
INTENSITY_RELATIVE_SCALE = 0.30
BBOX_RELATIVE_SCALE = 0.35
FEATURE_STUDENT_T_DOF = 4.0

# The scales above control tolerance; these weights temper the
# relative influence of evidence groups in the pair negative-log score.
INTERIOR_W_POSITION = 1.00
INTERIOR_W_VOLUME = 1.25
INTERIOR_W_SIZE = 0.35
INTERIOR_W_SHAPE = 0.55
INTERIOR_W_INTENSITY = 0.30
INTERIOR_W_BBOX = 0.20
INTERIOR_W_MOTION = 0.15

BOUNDARY_W_POSITION = 1.00
BOUNDARY_W_VOLUME = 0.10
BOUNDARY_W_MOTION = 0.25
BOUNDARY_W_INTENSITY = 0.25
BOUNDARY_W_FACE = 0.25

# ------------------------------------------------------------
# Explicit miss and birth priors
# ------------------------------------------------------------

# These are context priors used to construct -log(probability)
# alternatives in the augmented assignment matrix.
MISS_PROBABILITY_INTERIOR = 0.08
MISS_PROBABILITY_BOUNDARY = 0.30
MISS_PROBABILITY_PENDING_INTERIOR = 0.35
MISS_PROBABILITY_PENDING_BOUNDARY = 0.55
MISS_GLOBAL_UNCERTAINTY_BONUS = 0.15

BIRTH_PROBABILITY_INTERIOR = 0.06
BIRTH_PROBABILITY_BOUNDARY = 0.30

# Interior tracks receive one recovery frame after a selected miss.
INTERIOR_MAX_MISSING_FRAMES = 1
BOUNDARY_MAX_MISSING_FRAMES = 2

# ------------------------------------------------------------
# Broad physical safety gates
# ------------------------------------------------------------

# These are not operational association thresholds. They only remove
# physically absurd pairs after soft likelihoods have been computed.
ABSOLUTE_MAX_DISTANCE_INTERIOR_UM = 18.0
ABSOLUTE_MAX_DISTANCE_BOUNDARY_UM = 28.0
ABSOLUTE_DISTANCE_PER_MISSING_FRAME_UM = 5.0

ABSOLUTE_MAX_VOLUME_RATIO_INTERIOR = 8.0
ABSOLUTE_MAX_VOLUME_RATIO_BOUNDARY = 20.0

# ------------------------------------------------------------
# Boundary metadata and diagnostics
# ------------------------------------------------------------

BOUNDARY_MARGIN_UM = 2.0
TEMPLATE_EMA_ALPHA = 0.20

SAVE_TOP_ASSOCIATION_CANDIDATES = True
TOP_ASSOCIATION_CANDIDATES_PER_TRACK = 5

PROBABILITY_FLOOR = 1e-9
INVALID_COST = 1e6
EPS = 1e-8

