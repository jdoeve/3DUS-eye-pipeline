import numpy as np

from uceye.geometry import GeometryConfig
from uceye.recon import resample_to_cartesian


def test_resample_output_shape_and_finite():
    geom = GeometryConfig(
        image_height=64,
        r_max_pix=40.0,
        apex_xy=(8.0, 32.0),
        depth_mm=20.0,
        n_rows_mask=32,
        phi0_mask=0.0,
        phi1_mask=np.deg2rad(52.0),
    )
    unwrapped = np.ones((16, 40, 32), dtype=np.float32)
    theta_k = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False, dtype=np.float32)

    vol = resample_to_cartesian(
        unwrapped,
        geom,
        voxel_mm=1.0,
        sigma_theta_deg=0.0,
        theta_span_deg=360.0,
        interp_order=1,
        slab_size=8,
        theta_k=theta_k,
        verbose=False,
    )

    assert vol.shape == (40, 40, 40)
    assert np.isfinite(vol).all()
