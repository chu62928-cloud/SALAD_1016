"""Analytic checks for derivative order, units, boundaries and joint layout."""
import numpy as np
import torch


def verify_functions(upstream_jitter, diagnostics):
    checks = []
    fps = 20
    t = torch.arange(20, dtype=torch.float32) / fps
    p = torch.zeros(20, 22, 3)
    p[..., 0] = t[:, None] ** 3
    measured = float(upstream_jitter(p))
    assert abs(measured - 6) < .001, measured
    checks.append("cubic_mean_jerk_is_6_not_acceleration")
    q = p.clone()
    q[..., 1] = 2 * t[:, None] ** 3
    assert abs(float(upstream_jitter(q)) - 6 * np.sqrt(5)) < .002
    checks.append("Euclidean_norm_then_mean_not_RMS")
    assert abs(float(upstream_jitter(p * .00073)) - measured * .00073) < 1e-5
    checks.append("KIT_scale_is_linear")
    padded = torch.cat([p, torch.zeros_like(p)])
    assert float(upstream_jitter(padded)) > 10 * measured
    checks.append("padding_boundary_can_dominate")
    constant = np.zeros((32, 22, 3))
    constant[..., 0] = np.arange(22)[None, :] * .1
    result = diagnostics(constant, "t2m")
    assert max(abs(v) for k, v in result.items() if k not in ["toe_contact_interval_percent"]) < 1e-8
    checks.append("constant_pose_zero_temporal_metrics")
    shifted = constant + (np.arange(32) / 20)[:, None, None] * np.array([1, 0, 0])
    assert diagnostics(shifted, "t2m")["local_jerk_rms_mps3"] < 1e-8
    checks.append("root_relative_diagnostic_removes_translation")
    native_kit = constant[:, :21] * 1000
    kit = diagnostics(native_kit, "kit")
    assert kit["bone_cv_percent"] < 1e-8 and kit["world_jerk_rms_mps3"] == 0
    checks.append("KIT_21_joint_mm_native_fps_supported")
    return checks
