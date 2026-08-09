from ..adapters.nis3d import _spacing_from_info_text


def test_parser_uses_resolution_not_unrelated_intro_numbers():
    text = """Introduction:\nThe image is time-point 200 of embryo 4 and first 70 z-slices. The voxel size is 1 um x 1 um x 1 um.\nResolution:\n1 um x 1 um x 1 um\n"""
    assert _spacing_from_info_text(text) == (1.0, 1.0, 1.0)


def test_zebrafish_xyz_resolution_is_converted_to_zyx():
    text = "Resolution:\n0.43 um x 0.43 um x 2.5 um\n"
    assert _spacing_from_info_text(text) == (2.5, 0.43, 0.43)
