from .distance import (
    compute_distance_transform,
    smooth_distance_transform,
)

from .markers import (
    label_connected_components,
    detect_hmaxima,
    create_markers,
    ensure_component_markers,
)

from .watershed import watershed_segmentation


VOXEL_SIZE = (
    1.625,
    0.40625,
    0.40625,
)


def segment_instances(
        binary_mask,
):
    """
    Convert a binary foreground mask into instance labels.
    """

    distance = compute_distance_transform(
        binary_mask,
        voxel_size=VOXEL_SIZE,
    )

    distance = smooth_distance_transform(distance)

    component_labels, num_components = (
        label_connected_components(binary_mask)
    )

    hmax = detect_hmaxima(distance)

    markers, _ = create_markers(hmax)

    markers = ensure_component_markers(
        markers,
        component_labels,
        num_components,
        distance,
    )

    instance_labels = watershed_segmentation(
        distance,
        markers,
        binary_mask,
    )

    return instance_labels