"""Multiview bundle utilities for shared, distorted pinhole cameras.

The module keeps the small legacy helpers (``Layout``, ``residuals``,
``objective``, ``solve``) working while adding the shared-calibration layout,
the grouped Huber objective, layout observability analysis and a joint fit.

Geometry and objective contract
-------------------------------
* A calibration is identified by its ``calibration_id``.  All views that use
  that id share exactly one set of eight parameters; there is never a per-view
  copy and results are exported under the same id.
* A parameter named in ``calibration["fixed"]`` is not an optimization
  variable and its value is copied through bit-for-bit.
* ``view["fixed"]`` / ``point["fixed"]`` freeze the whole pose / xyz.
* For every observation ``e = (predicted_pixel - observed_pixel) / sigma``,
  ``q = ||e||_2`` and the cost is the Huber function of ``q`` (grouped over
  the two image coordinates), not a coordinate-wise Huber loss.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np

from camera import PARAMETERS, project

_SMALL_ANGLE = 1e-10


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------
def _skew(r):
    return np.array(
        [[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]],
        dtype=np.float64,
    )


def _skew_derivatives():
    # d [r]_x / d r_j
    out = []
    for j in range(3):
        d = np.zeros((3, 3), dtype=np.float64)
        if j == 0:
            d[1, 2] = -1.0
            d[2, 1] = 1.0
        elif j == 1:
            d[0, 2] = 1.0
            d[2, 0] = -1.0
        else:
            d[0, 1] = -1.0
            d[1, 0] = 1.0
        out.append(d)
    return out


_SKEW_DERIVATIVES = _skew_derivatives()


def rotation(rvec):
    """Axis-angle rotation matrix using the Rodrigues formula."""
    r = np.asarray(rvec, dtype=np.float64)
    theta = float(np.linalg.norm(r))
    K = _skew(r)
    if theta < _SMALL_ANGLE:
        return np.eye(3) + K + 0.5 * (K @ K)
    a = np.sin(theta) / theta
    b = (1.0 - np.cos(theta)) / (theta * theta)
    return np.eye(3) + a * K + b * (K @ K)


def rotation_derivatives(rvec):
    """Return R(rvec) and [dR/dr0, dR/dr1, dR/dr2]."""
    r = np.asarray(rvec, dtype=np.float64)
    theta = float(np.linalg.norm(r))
    K = _skew(r)
    K2 = K @ K
    if theta < _SMALL_ANGLE:
        R = np.eye(3) + K + 0.5 * K2
        derivs = []
        for j in range(3):
            dK = _SKEW_DERIVATIVES[j]
            derivs.append(dK + 0.5 * (dK @ K + K @ dK))
        return R, derivs

    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    a = sin_t / theta
    b = (1.0 - cos_t) / (theta * theta)
    da = (theta * cos_t - sin_t) / (theta * theta)
    db = (theta * sin_t - 2.0 * (1.0 - cos_t)) / (theta ** 3)
    R = np.eye(3) + a * K + b * K2
    dr = r / theta
    derivs = []
    for j in range(3):
        dK = _SKEW_DERIVATIVES[j]
        derivs.append(
            da * dr[j] * K
            + a * dK
            + db * dr[j] * K2
            + b * (dK @ K + K @ dK)
        )
    return R, derivs


# ---------------------------------------------------------------------------
# Layout: actual free variables and their owners
# ---------------------------------------------------------------------------
def _calibration_fixed(cal):
    fixed = cal.get("fixed", [])
    if fixed is None:
        return set()
    if isinstance(fixed, str):
        return {fixed}
    return {str(name) for name in fixed}


class Layout:
    """Free-variable layout for a scene.

    ``entries`` lists one tuple per free scalar.  Calibration variables are
    keyed by calibration id (so shared lenses have one group), not by view.
    """

    def __init__(self, scene):
        self.scene = deepcopy(scene)
        self._validate()

        self.entries = []          # identity tuples
        self.values = []           # initial values, matching entries
        self.cal_col = {}          # (cal_id, parameter) -> column
        self.view_col = {}         # (view_id, "rvec"|"t", coordinate) -> column
        self.point_col = {}        # (point_id, coordinate) -> column

        referenced = []
        for view in self.scene.get("views", []):
            cal_id = view["calibration_id"]
            if cal_id not in referenced:
                referenced.append(cal_id)

        # Calibration parameters, ordered by referenced calibration id.
        for cal_id in referenced:
            cal = self.scene["calibrations"][cal_id]
            fixed = _calibration_fixed(cal)
            for key in PARAMETERS:
                if key in fixed:
                    continue
                col = len(self.entries)
                self.cal_col[(cal_id, key)] = col
                self.entries.append(("calibration", cal_id, key))
                self.values.append(float(cal.get(key, 0.0)))

        # View poses.
        for view in self.scene["views"]:
            if view.get("fixed", False):
                continue
            for key in ("rvec", "t"):
                for j in range(3):
                    col = len(self.entries)
                    self.view_col[(view["id"], key, j)] = col
                    self.entries.append(("view", view["id"], key, j))
                    self.values.append(float(view[key][j]))

        # Points.
        for point in self.scene["points"]:
            if point.get("fixed", False):
                continue
            for j in range(3):
                col = len(self.entries)
                self.point_col[(point["id"], j)] = col
                self.entries.append(("point", point["id"], j))
                self.values.append(float(point["xyz"][j]))

        self.x0 = np.asarray(self.values, dtype=np.float64)
        self.nfree = len(self.entries)

    # -- validation ---------------------------------------------------------
    def _validate(self):
        scene = self.scene
        if not isinstance(scene, dict):
            raise ValueError("scene must be a mapping")
        for key in ("calibrations", "views", "points", "observations"):
            if key not in scene:
                raise ValueError("scene is missing %r" % (key,))
        if not isinstance(scene["calibrations"], dict):
            raise ValueError("calibrations must be a mapping keyed by id")

        view_ids = set()
        for view in scene["views"]:
            vid = view["id"]
            if vid in view_ids:
                raise ValueError("duplicate view id %r" % (vid,))
            view_ids.add(vid)
            cal_id = view["calibration_id"]
            if cal_id not in scene["calibrations"]:
                raise ValueError("view %r references unknown calibration %r" % (vid, cal_id))
            cal = scene["calibrations"][cal_id]
            for key in PARAMETERS:
                if key not in cal:
                    if key in ("fx", "fy", "cx", "cy"):
                        raise ValueError("calibration %r is missing %r" % (cal_id, key))
                    # Distortion parameters legitimately default to zero; do
                    # not write into the scene, only read them as 0.0.
                    continue
                value = float(cal[key])
                if not np.isfinite(value):
                    raise ValueError("calibration %r parameter %r must be finite" % (cal_id, key))
            if float(cal["fx"]) <= 0.0 or float(cal["fy"]) <= 0.0:
                raise ValueError("calibration %r has non-positive focal length" % (cal_id,))
            if len(view["rvec"]) != 3 or len(view["t"]) != 3:
                raise ValueError("view %r must have 3-vectors rvec and t" % (vid,))
            if not np.isfinite(np.asarray(view["rvec"], dtype=float)).all():
                raise ValueError("view %r rvec must be finite" % (vid,))
            if not np.isfinite(np.asarray(view["t"], dtype=float)).all():
                raise ValueError("view %r t must be finite" % (vid,))

        point_ids = set()
        for point in scene["points"]:
            pid = point["id"]
            if pid in point_ids:
                raise ValueError("duplicate point id %r" % (pid,))
            point_ids.add(pid)
            if len(point["xyz"]) != 3:
                raise ValueError("point %r must have a 3-vector xyz" % (pid,))
            if not np.isfinite(np.asarray(point["xyz"], dtype=float)).all():
                raise ValueError("point %r xyz must be finite" % (pid,))

        for obs in scene["observations"]:
            if obs["view_id"] not in view_ids:
                raise ValueError("observation references unknown view %r" % (obs["view_id"],))
            if obs["point_id"] not in point_ids:
                raise ValueError("observation references unknown point %r" % (obs["point_id"],))
            if len(obs["pixel"]) != 2:
                raise ValueError("observation pixel must have two coordinates")
            if not np.isfinite(np.asarray(obs["pixel"], dtype=float)).all():
                raise ValueError("observation pixel must be finite")
            sigma = float(obs.get("sigma", 1.0))
            if not np.isfinite(sigma) or sigma <= 0.0:
                raise ValueError("observation sigma must be positive and finite")

        # Keep convenience index maps; not part of the exported scene.
        self._view_index = {v["id"]: i for i, v in enumerate(scene["views"])}
        self._point_index = {p["id"]: i for i, p in enumerate(scene["points"])}

    # -- scene expansion ----------------------------------------------------
    def expand_scene(self, x):
        """Return the scene with free variables replaced by ``x``."""
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (self.nfree,):
            raise ValueError("x must have length %d" % self.nfree)
        scene = deepcopy(self.scene)
        for (kind, owner, *rest), value in zip(self.entries, x):
            value = float(value)
            if kind == "calibration":
                scene["calibrations"][owner][rest[0]] = value
            elif kind == "view":
                key, j = rest
                for view in scene["views"]:
                    if view["id"] == owner:
                        view[key][j] = value
                        break
            else:  # point
                j = rest[0]
                for point in scene["points"]:
                    if point["id"] == owner:
                        point["xyz"][j] = value
                        break
        return scene

    def expand(self, x):
        """Legacy-compatible expansion returning ``(scene, per-view cal list)``."""
        scene = self.expand_scene(x)
        view_cal = [scene["calibrations"][v["calibration_id"]] for v in scene["views"]]
        return scene, view_cal

    def result(self, x):
        return self.expand_scene(x)


def free_parameter_identities(layout):
    """Human/JSON-friendly identities of the actual free parameters."""
    out = []
    for kind, owner, *rest in layout.entries:
        if kind == "calibration":
            key = rest[0]
            out.append({
                "id": "calibration:%s:%s" % (owner, key),
                "kind": kind,
                "object": owner,
                "parameter": key,
                "index": None,
            })
        elif kind == "view":
            key, j = rest
            out.append({
                "id": "view:%s:%s:%d" % (owner, key, j),
                "kind": kind,
                "object": owner,
                "parameter": key,
                "index": int(j),
            })
        else:
            j = rest[0]
            out.append({
                "id": "point:%s:xyz:%d" % (owner, j),
                "kind": kind,
                "object": owner,
                "parameter": "xyz",
                "index": int(j),
            })
    return out


# ---------------------------------------------------------------------------
# Observation model, Jacobians and diagnostics
# ---------------------------------------------------------------------------
def _rotated_camera(view, xyz):
    R = rotation(view["rvec"])
    return R @ np.asarray(xyz, dtype=np.float64) + np.asarray(view["t"], dtype=np.float64)


def _observation_jacobian(layout, scene, view, point, obs, cam=None):
    """Return standardized d(e)/d(free variables), e and q for one obs."""
    cal_id = view["calibration_id"]
    cal = scene["calibrations"][cal_id]
    if cam is None:
        cam = _rotated_camera(view, point["xyz"])
    from camera import jacobian as _point_jacobian

    jp, jc = _point_jacobian(cam, cal)
    sigma = float(obs.get("sigma", 1.0))
    J = np.zeros((2, layout.nfree), dtype=np.float64)

    for k, key in enumerate(PARAMETERS):
        col = layout.cal_col.get((cal_id, key))
        if col is not None:
            J[:, col] = jc[:, k] / sigma

    if not view.get("fixed", False):
        R, dR = rotation_derivatives(view["rvec"])
        pw = np.asarray(point["xyz"], dtype=np.float64)
        for j in range(3):
            col = layout.view_col.get((view["id"], "rvec", j))
            if col is not None:
                J[:, col] = (jp @ (dR[j] @ pw)) / sigma
            col = layout.view_col.get((view["id"], "t", j))
            if col is not None:
                J[:, col] = jp[:, j] / sigma
    else:
        R = rotation(view["rvec"])

    if not point.get("fixed", False):
        for j in range(3):
            col = layout.point_col.get((point["id"], j))
            if col is not None:
                J[:, col] = (jp @ R[:, j]) / sigma

    pred = project(cam.reshape(1, 3), cal)[0]
    e = (pred - np.asarray(obs["pixel"], dtype=np.float64)) / sigma
    return J, e, float(np.linalg.norm(e))


def _huber_from_q(q, delta):
    if q <= delta:
        return 0.5 * q * q
    return delta * (q - 0.5 * delta)


def evaluate_scene(scene, delta=2.0):
    """Recompute residuals and cost directly from a scene.

    Returns ``{"cost", "rmse", "diagnostics", "all_visible"}``.  This is the
    authoritative path used by ``fit`` as well, so the exported scene can be
    checked independently.
    """
    delta = float(delta)
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("delta must be positive and finite")
    scene = deepcopy(scene)

    # Validate/set defaults through Layout without using free variables.
    layout = Layout(scene)
    scene = layout.scene
    view_index = {v["id"]: v for v in scene["views"]}
    point_index = {p["id"]: p for p in scene["points"]}

    diagnostics = []
    total = 0.0
    all_visible = True
    for obs in scene["observations"]:
        view = view_index[obs["view_id"]]
        point = point_index[obs["point_id"]]
        sigma = float(obs.get("sigma", 1.0))
        cam = _rotated_camera(view, point["xyz"])
        visible = bool(np.isfinite(cam).all() and cam[2] > 0.0)
        if visible:
            try:
                pred = project(cam.reshape(1, 3), scene["calibrations"][view["calibration_id"]])[0]
            except ValueError:
                visible = False
                pred = np.full(2, np.nan)
        else:
            pred = np.full(2, np.nan)

        if visible and np.isfinite(pred).all():
            e = (pred - np.asarray(obs["pixel"], dtype=np.float64)) / sigma
            q = float(np.linalg.norm(e))
            cost = _huber_from_q(q, delta)
            e_out = [float(e[0]), float(e[1])]
            pixel_residual = (pred - np.asarray(obs["pixel"], dtype=np.float64)).tolist()
            pred_out = pred.tolist()
        else:
            all_visible = False
            q = float("inf")
            cost = float("inf")
            e_out = [float("nan"), float("nan")]
            pixel_residual = [float("nan"), float("nan")]
            pred_out = [float("nan"), float("nan")]
        total += cost
        diagnostics.append({
            "view_id": obs["view_id"],
            "point_id": obs["point_id"],
            "sigma": sigma,
            "predicted_pixel": pred_out,
            "residual": e_out,
            "pixel_residual": pixel_residual,
            "q": q,
            "cost": cost,
            "visible": bool(visible),
        })

    visible_q = [d["q"] for d in diagnostics if d["visible"]]
    if visible_q:
        rmse = float(np.sqrt(np.mean(np.square(visible_q))))
        pixel_rmse = float(np.sqrt(np.mean([
            (d["pixel_residual"][0] ** 2 + d["pixel_residual"][1] ** 2)
            for d in diagnostics if d["visible"]
        ])))
    else:
        rmse = float("nan")
        pixel_rmse = float("nan")
    return {
        "cost": float(total),
        "rmse": rmse,
        "pixel_rmse": pixel_rmse,
        "diagnostics": diagnostics,
        "all_visible": bool(all_visible and np.isfinite(total)),
    }


# ---------------------------------------------------------------------------
# Layout analysis
# ---------------------------------------------------------------------------
def _jacobian_matrix(layout, scene=None):
    """Standardized, unrobustified reprojection Jacobian (2 rows per obs).

    Observations whose camera point has non-positive depth or non-finite
    values contribute zero rows; they cannot constrain the layout.  This keeps
    layout analysis usable even for a scene whose initial state violates the
    positive-depth domain.
    """
    if scene is None:
        scene = layout.scene
    view_index = {v["id"]: v for v in scene["views"]}
    point_index = {p["id"]: p for p in scene["points"]}
    J = np.zeros((2 * len(scene["observations"]), layout.nfree), dtype=np.float64)
    row = 0
    for obs in scene["observations"]:
        view = view_index[obs["view_id"]]
        point = point_index[obs["point_id"]]
        cam = _rotated_camera(view, point["xyz"])
        if np.isfinite(cam).all() and cam[2] > 0.0:
            try:
                J2, _, _ = _observation_jacobian(layout, scene, view, point, obs, cam=cam)
            except ValueError:
                J2 = None
            if J2 is not None and np.isfinite(J2).all():
                J[row:row + 2, :] = J2
        row += 2
    return J


def _normalized_rank(J):
    n = J.shape[1]
    if n == 0 or J.size == 0:
        return 0, np.zeros(0)
    norms = np.linalg.norm(J, axis=0)
    Jn = np.zeros_like(J)
    nonzero = norms > 0.0
    Jn[:, nonzero] = J[:, nonzero] / norms[nonzero]
    s = np.linalg.svd(Jn, compute_uv=False)
    s = np.asarray(s, dtype=np.float64)
    if s.size == 0 or s[0] <= 0.0:
        return 0, s
    rank = int(np.count_nonzero(s > s[0] * 1e-7))
    return rank, s


def analyze_layout(scene):
    """Report free parameter identities, ownership and Jacobian rank.

    Rank uses the unrobustified standardized reprojection Jacobian with each
    nonzero column normalized to unit 2-norm (zero columns stay zero) and a
    singular-value threshold of ``1e-7 * max(singular value)``.
    """
    layout = Layout(scene)
    J = _jacobian_matrix(layout)
    rank, singular_values = _normalized_rank(J)
    norms = np.linalg.norm(J, axis=0)
    return {
        "free_parameters": free_parameter_identities(layout),
        "free_parameter_count": int(layout.nfree),
        "layout_rank": int(rank),
        "missing_dimensions": int(max(0, layout.nfree - rank)),
        "observable": bool(rank == layout.nfree),
        "column_norms": [float(v) for v in norms],
        "singular_values": [float(v) for v in singular_values],
        "observation_count": len(layout.scene["observations"]),
    }


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------
def _huber_cost(q, delta):
    q = np.asarray(q, dtype=np.float64)
    return np.sum(np.where(q <= delta, 0.5 * q * q, delta * (q - 0.5 * delta)))


def _linearize(layout, x):
    """Return grouped residuals E (n,2), Jacobians J (n,2,p) and validity.

    ``E`` holds standardized 2-D residuals, ``J`` their derivatives with
    respect to the free variables.  Validity is False if any observation has
    non-finite values or non-positive camera depth.
    """
    scene = layout.expand_scene(x)
    view_index = {v["id"]: v for v in scene["views"]}
    point_index = {p["id"]: p for p in scene["points"]}
    observations = scene["observations"]
    nobs = len(observations)
    nfree = layout.nfree
    E = np.zeros((nobs, 2), dtype=np.float64)
    J = np.zeros((nobs, 2, nfree), dtype=np.float64)
    valid = True
    for i, obs in enumerate(observations):
        view = view_index[obs["view_id"]]
        point = point_index[obs["point_id"]]
        cam = _rotated_camera(view, point["xyz"])
        if not np.isfinite(cam).all() or cam[2] <= 0.0:
            valid = False
            continue
        try:
            pred = project(cam.reshape(1, 3), scene["calibrations"][view["calibration_id"]])[0]
        except ValueError:
            valid = False
            continue
        if not np.isfinite(pred).all():
            valid = False
            continue
        E[i, :] = (pred - np.asarray(obs["pixel"], dtype=np.float64)) / float(obs["sigma"])
        J[i, :, :] = _observation_jacobian(layout, scene, view, point, obs, cam=cam)[0]
    q = np.linalg.norm(E, axis=1)
    return E, J, q, valid


def _state_cost(E, q, delta):
    if not np.isfinite(E).all() or not np.isfinite(q).all():
        return float("inf")
    return float(_huber_cost(q, delta))


def _optimize(layout, delta, max_iter=200):
    """Levenberg-Marquardt on the grouped Huber cost with analytic Jacobians.

    The robust loss is handled with iteratively reweighted Gauss-Newton where
    each observation contributes a scalar Huber weight times its full 2-D
    residual block (never a per-coordinate decision).  Variables are scaled by
    the Jacobian column norms so fx/fy and the distortion terms are not
    dwarfed by pixel-space translation columns.
    """
    x = np.asarray(layout.x0, dtype=np.float64).copy()
    nfree = layout.nfree

    E, J, q, valid = _linearize(layout, x)
    if not valid:
        return x, E, q, False, 0, "initial state leaves the valid camera domain"
    cost = _state_cost(E, q, delta)
    initial_cost = cost

    stacked = J.reshape(2 * len(q), nfree)
    col_norms = np.linalg.norm(stacked, axis=0)
    if col_norms.size:
        floor = max(1e-8 * col_norms.max(), 1e-12)
        scale = 1.0 / np.maximum(col_norms, floor)
    else:
        scale = np.ones(0)

    mu = 1e-4
    max_mu = 1e20
    converged = False
    message = "maximum iterations reached"
    iterations = 0

    for iterations in range(1, max_iter + 1):
        active = q <= delta
        weights = np.where(active, 1.0, delta / np.maximum(q, 1e-300))
        WE = weights[:, None] * E
        grad = np.einsum("nij,ni->j", J, WE)
        H = np.einsum("nij,niq,n->jq", J, J, weights)
        Hs = (scale[:, None] * H) * scale[None, :]
        gs = scale * grad
        diag = np.maximum(np.diag(Hs), 1e-16)

        # Convergence: scaled gradient is numerically zero.
        if np.max(np.abs(gs), initial=0.0) <= 1e-10 * max(1.0, abs(initial_cost)):
            converged = True
            message = "gradient tolerance reached"
            break

        improved = False
        while mu <= max_mu:
            A = Hs + mu * np.diag(diag)
            try:
                dy = np.linalg.solve(A, -gs)
            except np.linalg.LinAlgError:
                mu *= 10.0
                continue
            if not np.isfinite(dy).all():
                mu *= 10.0
                continue
            x_new = x + scale * dy
            E_new, J_new, q_new, valid_new = _linearize(layout, x_new)
            if valid_new:
                cost_new = _state_cost(E_new, q_new, delta)
            else:
                cost_new = float("inf")
            if np.isfinite(cost_new) and cost_new < cost:
                relative = (cost - cost_new) / max(cost, 1e-300)
                x, E, J, q, cost = x_new, E_new, J_new, q_new, cost_new
                mu = max(mu / 3.0, 1e-15)
                improved = True
                if relative <= 1e-14:
                    converged = True
                    message = "relative cost tolerance reached"
                break
            mu *= 4.0
        if converged:
            break
        if not improved:
            # Damping could not find a reduction.  If the scaled gradient is
            # already negligible, this is a numerical optimum rather than a
            # stall; otherwise report an honest non-convergence.
            active = q <= delta
            weights = np.where(active, 1.0, delta / np.maximum(q, 1e-300))
            grad = np.einsum("nij,ni->j", J, weights[:, None] * E)
            gs = scale * grad
            if np.max(np.abs(gs), initial=0.0) <= 1e-8 * max(1.0, abs(initial_cost)):
                converged = True
                message = "gradient tolerance reached at damping floor"
            else:
                message = "no cost-reducing step from Levenberg-Marquardt damping"
            break

    return x, E, q, True, iterations, message


def fit(scene, delta=2.0):
    """Jointly fit free parameters of a shared-calibration scene.

    Parameters that are not free are copied unchanged.  If the layout Jacobian
    is rank deficient the function returns ``status="unobservable"`` without
    inventing a reconstruction.  Otherwise a local Levenberg-Marquardt solve of
    the grouped Huber objective is attempted; a solver that does not converge
    honestly reports ``status="not_converged"``.
    """
    delta = float(delta)
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("delta must be positive and finite")

    base_scene = deepcopy(scene)
    layout = Layout(base_scene)
    nfree = int(layout.nfree)

    initial = evaluate_scene(layout.scene, delta)
    initial_cost = float(initial["cost"])
    J = _jacobian_matrix(layout)
    rank, _ = _normalized_rank(J)

    common = {
        "scene": deepcopy(layout.scene),
        "initial_cost": initial_cost,
        "layout_rank": int(rank),
        "free_parameter_count": nfree,
        "delta": delta,
    }

    if not np.isfinite(initial_cost):
        # The objective is undefined (e.g. an observation behind the camera at
        # the initial values).  Rank analysis would be meaningless here, so do
        # not report a layout result or crash; report failure honestly.
        common.update({
            "status": "not_converged",
            "final_cost": initial_cost,
            "missing_dimensions": 0,
            "diagnostics": initial["diagnostics"],
            "all_visible": False,
            "pixel_rmse": initial["pixel_rmse"],
            "standardized_rmse": initial["rmse"],
            "message": "initial state is outside the valid camera domain",
        })
        return common

    if rank < nfree:
        common.update({
            "status": "unobservable",
            "final_cost": initial_cost,
            "missing_dimensions": int(nfree - rank),
            "diagnostics": initial["diagnostics"],
            "all_visible": initial["all_visible"],
            "pixel_rmse": initial["pixel_rmse"],
            "standardized_rmse": initial["rmse"],
            "message": "layout rank %d is smaller than free parameter count %d" % (rank, nfree),
        })
        return common

    if nfree == 0 or len(layout.scene["observations"]) == 0:
        common.update({
            "status": "converged",
            "final_cost": initial_cost,
            "missing_dimensions": 0,
            "diagnostics": initial["diagnostics"],
            "all_visible": initial["all_visible"],
            "pixel_rmse": initial["pixel_rmse"],
            "standardized_rmse": initial["rmse"],
            "message": "no free variables",
        })
        return common

    x, E, q, valid, iterations, message = _optimize(layout, delta)
    result_scene = layout.expand_scene(x)
    final = evaluate_scene(result_scene, delta)
    final_cost = float(final["cost"])

    # Accepting a solution requires a valid, non-increasing state.  Never
    # present an invalid or regressed reconstruction as a fit.
    accepted = bool(
        valid
        and np.isfinite(final_cost)
        and final["all_visible"]
        and final_cost <= initial_cost + 1e-9 * max(1.0, abs(initial_cost))
    )
    converged = bool(accepted and message.startswith(("gradient", "relative")))
    if not accepted:
        result_scene = deepcopy(layout.scene)
        final = initial
        final_cost = initial_cost

    active = np.array([d["visible"] and d["q"] <= delta for d in final["diagnostics"]], dtype=bool)
    common.update({
        "status": "converged" if converged else "not_converged",
        "scene": result_scene,
        "final_cost": final_cost,
        "missing_dimensions": 0,
        "diagnostics": final["diagnostics"],
        "all_visible": final["all_visible"],
        "pixel_rmse": final["pixel_rmse"],
        "standardized_rmse": final["rmse"],
        "message": message,
        "iterations": int(iterations),
        "inlier_count": int(np.count_nonzero(active)),
        "outlier_count": int(np.count_nonzero(~active)),
        "free_parameters": free_parameter_identities(layout),
    })
    return common


# ---------------------------------------------------------------------------
# Legacy-compatible helpers
# ---------------------------------------------------------------------------
def residuals(layout, x):
    """Raw standardized residuals (2 per observation), flattened.

    Kept for the original independent-pinhole entry point.  No robust loss is
    applied here; use :func:`evaluate_scene` for grouped Huber diagnostics.
    """
    scene = layout.expand_scene(x) if hasattr(layout, "expand_scene") else layout.expand(x)[0]
    view_index = {v["id"]: v for v in scene["views"]}
    point_index = {p["id"]: p for p in scene["points"]}
    out = []
    for obs in scene["observations"]:
        view = view_index[obs["view_id"]]
        point = point_index[obs["point_id"]]
        cam = _rotated_camera(view, point["xyz"])
        pred = project(cam.reshape(1, 3), scene["calibrations"][view["calibration_id"]])[0]
        out.extend((pred - np.asarray(obs["pixel"], dtype=np.float64)) / float(obs.get("sigma", 1.0)))
    return np.asarray(out, dtype=np.float64)


def objective(layout, x, delta=2.0):
    """Grouped Huber objective at ``x`` using the layout's scene."""
    scene = layout.expand_scene(x) if hasattr(layout, "expand_scene") else layout.expand(x)[0]
    return evaluate_scene(scene, delta)["cost"]


def solve(scene, delta=2.0):
    """Legacy entry point returning the same shape of result as before.

    ``status`` is ``converged``, ``not_converged`` or ``unobservable`` and
    ``cost`` is the final grouped-Huber cost.  The richer fields from
    :func:`fit` are included as well.
    """
    result = fit(scene, delta=delta)
    out = dict(result)
    out["cost"] = out["final_cost"]
    return out
