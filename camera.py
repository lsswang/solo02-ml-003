"""Legacy pinhole geometry used by the scene fitter and standalone pixel clients."""
import numpy as np

PARAMETERS = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2")


def project(points_camera, calibration):
    points = np.asarray(points_camera, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("camera points must be finite N x 3")
    if np.any(points[:, 2] <= 0):
        raise ValueError("points must have positive camera depth")
    xy = points[:, :2] / points[:, 2:3]
    return xy * [calibration["fx"], calibration["fy"]] + [calibration["cx"], calibration["cy"]]


def unproject(pixels, calibration):
    pixels = np.asarray(pixels, dtype=np.float64)
    xy = (pixels - [calibration["cx"], calibration["cy"]]) / [calibration["fx"], calibration["fy"]]
    return np.column_stack([xy, np.ones(len(xy))])


def jacobian(point_camera, calibration, step=1e-6):
    """Return d(pixel)/d(X,Y,Z) and d(pixel)/d(PARAMETERS), holding other values fixed."""
    point = np.asarray(point_camera, dtype=np.float64)
    jp = np.empty((2, 3))
    jc = np.empty((2, len(PARAMETERS)))
    for j in range(3):
        delta = np.zeros(3); delta[j] = step
        jp[:, j] = (project([point + delta], calibration)[0] - project([point - delta], calibration)[0]) / (2 * step)
    for j, name in enumerate(PARAMETERS):
        a = dict(calibration); b = dict(calibration)
        a[name] = a.get(name, 0.0) + step; b[name] = b.get(name, 0.0) - step
        jc[:, j] = (project([point], a)[0] - project([point], b)[0]) / (2 * step)
    return jp, jc
