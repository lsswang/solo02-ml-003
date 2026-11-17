"""Pinhole camera geometry with radial and tangential distortion.

The public functions are intentionally small and directly usable outside the
bundle adjuster:

    project(points_camera, calibration) -> (N, 2) pixels
    unproject(pixels, calibration)      -> (N, 3) rays with Z == 1
    jacobian(point_camera, calibration) -> (d pixel / d X, d pixel / d params)

Camera coordinates use positive Z forward.  With x = X/Z and y = Y/Z and
r2 = x*x + y*y the distortion model is

    xd = x (1 + k1 r2 + k2 r2**2) + 2 p1 x y + p2 (r2 + 2 x*x)
    yd = y (1 + k1 r2 + k2 r2**2) + p1 (r2 + 2 y*y) + 2 p2 x y

and pixels are (fx*xd + cx, fy*yd + cy).  Missing distortion keys default to
zero so the legacy zero-distortion calls are unchanged.
"""
from __future__ import annotations

import numpy as np

PARAMETERS = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2")


def _calibration_values(calibration):
    """Validate a calibration and return the eight parameters as floats."""
    values = []
    for name in PARAMETERS:
        try:
            value = calibration[name]
        except (KeyError, TypeError):
            if name in ("fx", "fy", "cx", "cy"):
                raise ValueError("calibration is missing %r" % (name,)) from None
            value = 0.0
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("calibration parameter %r must be finite" % (name,))
        values.append(value)
    fx, fy = values[0], values[1]
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("focal lengths must be positive")
    return values


def _distort_xy(x, y, k1, k2, p1, p2):
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return xd, yd, r2, radial


def _distort_jacobian_xy(x, y, k1, k2, p1, p2):
    """Derivatives of the normalized distortion map with respect to (x, y)."""
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2
    dradial = k1 + 2.0 * k2 * r2  # d radial / d r2
    # d xd / d x, d xd / d y
    dxd_dx = radial + 2.0 * x * x * dradial + 2.0 * p1 * y + 6.0 * p2 * x
    dxd_dy = 2.0 * x * y * dradial + 2.0 * p1 * x + 2.0 * p2 * y
    # d yd / d x, d yd / d y
    dyd_dx = 2.0 * x * y * dradial + 2.0 * p1 * x + 2.0 * p2 * y
    dyd_dy = radial + 2.0 * y * y * dradial + 6.0 * p1 * y + 2.0 * p2 * x
    return np.array([[dxd_dx, dxd_dy], [dyd_dx, dyd_dy]], dtype=np.float64)


def project(points_camera, calibration):
    """Project finite N x 3 camera points (positive Z) to N x 2 pixels."""
    points = np.asarray(points_camera, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("camera points must be a finite N x 3 array")
    if not np.isfinite(points).all():
        raise ValueError("camera points must be finite")
    if np.any(points[:, 2] <= 0.0):
        raise ValueError("points must have positive camera depth")

    fx, fy, cx, cy, k1, k2, p1, p2 = _calibration_values(calibration)
    x = points[:, 0] / points[:, 2]
    y = points[:, 1] / points[:, 2]
    xd, yd, _, _ = _distort_xy(x, y, k1, k2, p1, p2)
    out = np.empty((points.shape[0], 2), dtype=np.float64)
    out[:, 0] = fx * xd + cx
    out[:, 1] = fy * yd + cy
    return out


def unproject(pixels, calibration):
    """Map pixels to undistorted camera rays with Z == 1.

    The returned point (x, y, 1) satisfies project([(x, y, 1)], calibration)
    == pixel up to numerical precision.
    """
    pixels = np.asarray(pixels, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("pixels must be a finite N x 2 array")
    if not np.isfinite(pixels).all():
        raise ValueError("pixels must be finite")

    fx, fy, cx, cy, k1, k2, p1, p2 = _calibration_values(calibration)
    xd = (pixels[:, 0] - cx) / fx
    yd = (pixels[:, 1] - cy) / fy

    rays = np.empty((pixels.shape[0], 3), dtype=np.float64)
    for i in range(pixels.shape[0]):
        # Newton solve for the undistorted normalized coordinates.  The
        # requested domain is the small well-behaved region in front of the
        # camera; no attempt is made to enumerate roots outside it.
        x = xd[i]
        y = yd[i]
        for _ in range(100):
            px, py, _, _ = _distort_xy(x, y, k1, k2, p1, p2)
            ex = px - xd[i]
            ey = py - yd[i]
            if ex == 0.0 and ey == 0.0:
                break
            jac = _distort_jacobian_xy(x, y, k1, k2, p1, p2)
            try:
                step = np.linalg.solve(jac, np.array([ex, ey], dtype=np.float64))
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(jac, np.array([ex, ey]), rcond=None)[0]
            x -= step[0]
            y -= step[1]
            if max(abs(step[0]), abs(step[1])) <= 1e-15:
                break
        rays[i, 0] = x
        rays[i, 1] = y
        rays[i, 2] = 1.0
    return rays


def jacobian(point_camera, calibration, step=1e-6):
    """Return d(pixel)/d(X,Y,Z) and d(pixel)/d(PARAMETERS).

    The derivatives are analytic and evaluated with all other quantities held
    fixed.  ``step`` is accepted for backwards compatibility with the legacy
    finite-difference signature and is not used.
    """
    point = np.asarray(point_camera, dtype=np.float64)
    if point.ndim == 2 and point.shape == (1, 3):
        point = point[0]
    if point.shape != (3,):
        raise ValueError("point_camera must be a 3-vector")
    if not np.isfinite(point).all():
        raise ValueError("camera point must be finite")
    if point[2] <= 0.0:
        raise ValueError("point must have positive camera depth")

    fx, fy, cx, cy, k1, k2, p1, p2 = _calibration_values(calibration)
    X, Y, Z = point
    x = X / Z
    y = Y / Z
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2
    dradial = k1 + 2.0 * k2 * r2

    # Jacobian of the normalized distortion map.
    D = _distort_jacobian_xy(x, y, k1, k2, p1, p2)

    # d(x, y) / d(X, Y, Z)
    dxy_dcam = np.array(
        [[1.0 / Z, 0.0, -x / Z],
         [0.0, 1.0 / Z, -y / Z]],
        dtype=np.float64,
    )
    # d(xd, yd) / d(X, Y, Z)
    dnorm_dcam = D @ dxy_dcam
    jp = np.empty((2, 3), dtype=np.float64)
    jp[0, :] = fx * dnorm_dcam[0, :]
    jp[1, :] = fy * dnorm_dcam[1, :]

    xd, yd, _, _ = _distort_xy(x, y, k1, k2, p1, p2)
    jc = np.zeros((2, len(PARAMETERS)), dtype=np.float64)
    # fx, fy, cx, cy
    jc[0, 0] = xd
    jc[1, 1] = yd
    jc[0, 2] = 1.0
    jc[1, 3] = 1.0
    # k1, k2
    jc[0, 4] = fx * x * r2
    jc[1, 4] = fy * y * r2
    jc[0, 5] = fx * x * r2 * r2
    jc[1, 5] = fy * y * r2 * r2
    # p1, p2
    jc[0, 6] = fx * 2.0 * x * y
    jc[1, 6] = fy * (r2 + 2.0 * y * y)
    jc[0, 7] = fx * (r2 + 2.0 * x * x)
    jc[1, 7] = fy * 2.0 * x * y
    return jp, jc
