import numpy as np

from uceye.geometry import GeometryConfig


def test_geometry_uses_mask_rows_for_delta_phi():
    geom = GeometryConfig(
        image_height=512,
        r_max_pix=400.0,
        apex_xy=(32.0, 256.0),
        depth_mm=48.0,
        n_rows_mask=256,
        phi0_mask=0.0,
        phi1_mask=np.deg2rad(52.0),
    )
    expected = np.deg2rad(52.0) / 255.0
    assert np.isclose(geom.delta_phi_per_px, expected)


def test_geometry_dr_mm():
    geom = GeometryConfig(image_height=512, r_max_pix=480.0, apex_xy=(0.0, 0.0), depth_mm=48.0)
    assert np.isclose(geom.dr_mm, 0.1)
