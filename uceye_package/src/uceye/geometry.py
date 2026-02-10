"""Geometry configuration and derived measurement helpers."""

import numpy as np

from .config import EYE_CUBED_DEPTH_MM, EYE_CUBED_FAN_ANGLE_DEG, FAN_ANGLE_RAD, SOS_SCALE


class GeometryConfig:
    """Centralized geometry configuration for unwrapping and reconstruction."""

    def __init__(
        self,
        image_height: int,
        r_max_pix: float,
        apex_xy: tuple,
        depth_mm: float = None,
        n_rows_mask: int = None,
        phi0_mask: float | None = None,
        phi1_mask: float | None = None,
    ):
        self.image_height = image_height
        self.r_max_pix = r_max_pix
        self.apex_xy = apex_xy
        self.image_height_eff = n_rows_mask if n_rows_mask is not None else image_height

        self.depth_mm_displayed = depth_mm if depth_mm else EYE_CUBED_DEPTH_MM
        self.depth_mm_sos_corrected = self.depth_mm_displayed * SOS_SCALE

        spec_rad = FAN_ANGLE_RAD
        use_mask = (phi0_mask is not None) and (phi1_mask is not None)

        if use_mask:
            lo = float(phi0_mask)
            hi = float(phi1_mask)
            if hi < lo:
                lo, hi = hi, lo
            if lo < 0.0:
                offset = -lo
                lo += offset
                hi += offset
            self.phi0 = lo
            self.phi1 = hi
            self.fan_angle_rad = self.phi1 - self.phi0
            self.fan_angle_deg = float(np.degrees(self.fan_angle_rad))
            fan_src = "mask"
        else:
            self.fan_angle_rad = spec_rad
            self.fan_angle_deg = EYE_CUBED_FAN_ANGLE_DEG
            self.phi0 = 0.0
            self.phi1 = self.fan_angle_rad
            fan_src = "spec"

        if self.image_height_eff > 1:
            self.delta_phi_per_px = self.fan_angle_rad / float(self.image_height_eff - 1)
        else:
            self.delta_phi_per_px = self.fan_angle_rad

        self.dr_mm = self.depth_mm_sos_corrected / r_max_pix

        print(
            f"[GEOM] fan src={fan_src}, fan_deg={self.fan_angle_deg:.3f}, "
            f"Δφ={self.delta_phi_per_px:.6f} rad/px, dr={self.dr_mm:.5f} mm/px, "
            f"rows_eff={self.image_height_eff}"
        )

    def get_arc_length_mm(self, radius_mm: float) -> float:
        return radius_mm * self.fan_angle_rad

    def get_lateral_mm_per_px(self, radius_mm: float) -> float:
        return radius_mm * self.delta_phi_per_px

    def print_summary(self):
        print("\n" + "=" * 60)
        print("GEOMETRY CONFIGURATION (SINGLE SOURCE OF TRUTH)")
        print("=" * 60)
        print(f"Fan angle:        {self.fan_angle_deg:.1f}° (device spec, LOCKED)")
        print(
            f"Δφ per pixel:     {self.delta_phi_per_px:.6f} rad/px = "
            f"{np.degrees(self.delta_phi_per_px):.4f}°/px"
        )
        print(f"Depth (display):  {self.depth_mm_displayed:.2f} mm")
        print(f"Depth (SoS corr): {self.depth_mm_sos_corrected:.2f} mm")
        print(f"Radial dr:        {self.dr_mm:.4f} mm/px (constant)")
        print(f"Apex (x, y):      ({self.apex_xy[0]:.1f}, {self.apex_xy[1]:.1f}) px")
        print(f"r_max:            {self.r_max_pix:.1f} px")
        print(f"Image height:     {self.image_height} px (full image)")
        print(f"Effective rows:   {self.image_height_eff} px (usable mask span)")
        print("\nLateral resolution (varies with depth):")
        for r in [10, 20, 30, 40]:
            print(f"  r={r:2d}mm: {self.get_lateral_mm_per_px(r):.4f} mm/px")
        print("=" * 60 + "\n")
