"""Small legacy bundle fitter: per-view pinhole calibration, fixed poses/points supported."""
from copy import deepcopy
import numpy as np
from camera import PARAMETERS, project


def rotation(rvec):
    r = np.asarray(rvec, dtype=np.float64)
    theta = np.linalg.norm(r)
    skew = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
    if theta < 1e-12:
        return np.eye(3) + skew
    return np.eye(3) + np.sin(theta) / theta * skew + (1 - np.cos(theta)) / theta ** 2 * (skew @ skew)


class Layout:
    """Legacy independent-calibration layout; each view contributes its own intrinsics."""
    def __init__(self, scene):
        self.scene = deepcopy(scene)
        self.entries = []
        self.values = []
        for vi, view in enumerate(scene["views"]):
            cal = scene["calibrations"][view["calibration_id"]]
            for key in PARAMETERS[:4]:
                self.entries.append(("calibration", vi, key)); self.values.append(cal[key])
            if not view.get("fixed", False):
                for key in ("rvec", "t"):
                    for j, value in enumerate(view[key]):
                        self.entries.append((key, vi, j)); self.values.append(value)
        for pi, point in enumerate(scene["points"]):
            if not point.get("fixed", False):
                for j, value in enumerate(point["xyz"]):
                    self.entries.append(("point", pi, j)); self.values.append(value)
        self.x0 = np.asarray(self.values, dtype=np.float64)

    def expand(self, x):
        scene = deepcopy(self.scene)
        view_cal = [deepcopy(scene["calibrations"][v["calibration_id"]]) for v in scene["views"]]
        for (kind, owner, key), value in zip(self.entries, x):
            if kind == "calibration":
                view_cal[owner][key] = float(value)
            elif kind == "point":
                scene["points"][owner]["xyz"][key] = float(value)
            else:
                scene["views"][owner][kind][key] = float(value)
        return scene, view_cal

    def result(self, x):
        scene, view_cal = self.expand(x)
        for view, cal in zip(scene["views"], view_cal):
            scene["calibrations"][view["calibration_id"]] = cal
        return scene


def residuals(layout, x):
    scene, view_cal = layout.expand(x)
    view_index = {v["id"]: i for i, v in enumerate(scene["views"])}
    point_index = {p["id"]: i for i, p in enumerate(scene["points"])}
    result = []
    for obs in scene["observations"]:
        vi = view_index[obs["view_id"]]
        view = scene["views"][vi]
        point = scene["points"][point_index[obs["point_id"]]]["xyz"]
        cam = rotation(view["rvec"]) @ point + view["t"]
        pixel = project([cam], view_cal[vi])[0]
        result.extend((pixel - obs["pixel"]) / obs.get("sigma", 1.0))
    return np.asarray(result)


def objective(layout, x, delta=2.0):
    # Legacy least-squares Huber convention applies the scalar loss to each coordinate.
    r = np.abs(residuals(layout, x))
    return float(np.sum(np.where(r <= delta, 0.5 * r * r, delta * (r - 0.5 * delta))))


def solve(scene, delta=2.0):
    from scipy.optimize import least_squares
    layout = Layout(scene)
    if len(layout.x0) == 0:
        return {"status": "converged", "scene": scene, "cost": objective(layout, layout.x0, delta)}
    fit = least_squares(lambda x: residuals(layout, x), layout.x0, loss="huber", f_scale=delta, max_nfev=300)
    return {"status": "converged" if fit.success else "not_converged", "scene": layout.result(fit.x), "cost": objective(layout, fit.x, delta)}
