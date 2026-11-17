# Small multiview geometry with shared, distorted lenses

Python 3.11. Install `requirements.txt`, then run the tests and demo:

```bash
pip install -r requirements.txt
python -m pytest -q
python demo.py                 # legacy independent-pinhole scene
python demo.py shared_clean    # shared-lens joint reconstruction
```

## `camera.py` — standalone geometry

Directly callable interfaces (not bundle-only helpers):

- `project(points_camera, calibration)` — N×3 camera points (positive Z) to N×2
  pixels, with radial/tangential distortion.
- `unproject(pixels, calibration)` — pixels to undistorted rays with `Z = 1`.
- `jacobian(point_camera, calibration)` — analytic 2×3 `d(pixel)/d(X,Y,Z)` and
  2×8 `d(pixel)/d(fx,fy,cx,cy,k1,k2,p1,p2)`.

Model: `X/Z = x`, `Y/Z = y`, `r² = x² + y²`, parameter order
`fx, fy, cx, cy, k1, k2, p1, p2`, pixels
`(fx·x' + cx, fy·y' + cy)` with

```
x' = x(1 + k1 r² + k2 r⁴) + 2 p1 x y + p2(r² + 2x²)
y' = y(1 + k1 r² + k2 r⁴) + p1(r² + 2y²) + 2 p2 x y
```

Missing distortion keys default to zero, so legacy zero-distortion calls are
unchanged. Non-finite values, non-positive focal lengths and non-positive
depths raise `ValueError`.

## Scene format

```json
{
  "calibrations": {"lens": {"fx": 528, "...": "...", "fixed": ["cx", "cy"]}},
  "views":  [{"id": "a", "calibration_id": "lens", "rvec": [...], "t": [...], "fixed": true}],
  "points": [{"id": "p00", "xyz": [...], "fixed": true}],
  "observations": [{"view_id": "a", "point_id": "p00", "pixel": [u, v], "sigma": 1.0}]
}
```

World points map to camera points via `R(rvec) @ xyz + t` (axis-angle `rvec`).
A `calibration_id` names exactly one parameter group, shared by every view that
references it. `calibration["fixed"]` freezes individual parameters; a view or
point with `"fixed": true` freezes the whole pose / xyz.

## `bundle.py` — layout, objective and fit

- `Layout(scene)` — the actual free-variable layout. Calibration columns are
  keyed by `(calibration_id, parameter)`, so a shared lens contributes one
  group, never per-view copies. `free_parameter_identities(layout)` lists every
  free scalar with its owning object and parameter.
- `analyze_layout(scene)` — free parameter identities, per-item ownership,
  column norms and the numerical rank of the unrobustified, sigma-standardized
  reprojection Jacobian. Nonzero columns are normalized to unit 2-norm (zero
  columns stay zero); the rank threshold is `1e-7 × max singular value`.
- `evaluate_scene(scene, delta=2.0)` — recomputes the grouped Huber objective
  and per-observation diagnostics directly from a scene. This is the
  authoritative path, so the exported scene can be rechecked independently.
- `fit(scene, delta=2.0)` — joint local fit. Returns `status`
  (`converged` / `unobservable` / `not_converged`), `scene`, `initial_cost`,
  `final_cost`, `layout_rank`, `free_parameter_count`, `missing_dimensions`,
  per-observation `diagnostics` (`residual`, `q`, `cost`, `visible`, …) and
  inlier/outlier counts.
- `solve`, `residuals`, `objective` — legacy-compatible entry points.

Objective per observation: `e = (predicted - observed) / sigma`, `q = ||e||₂`,
cost `q²/2` if `q ≤ δ` else `δ(q − δ/2)`. The Huber decision is made on the
2-D norm, not per coordinate.

## Derivatives and solver

Analytic derivatives are used throughout: `camera.jacobian` provides pixel
derivatives, and `bundle.rotation_derivatives` provides `dR/drvec`. The solver
is Levenberg–Marquardt on the grouped Huber cost with iteratively reweighted
Gauss–Newton (one scalar Huber weight per observation applied to its full 2-D
residual block). Variables are scaled by the Jacobian column norms. Typical
cost: well under 0.1 s for the five-view / 120-observation budget.

## Verification

`tests/test_geometry_and_bundle.py` checks geometry round trips (`≤2e-8` ray,
`≤2e-6` px projection), analytic-vs-finite-difference Jacobians
(`atol=rtol=2e-5`), shared/locked parameter identity, grouped (not
per-coordinate) Huber cost, independent recomputation of the exported cost and
residuals, rank-deficient and invalid-domain reporting, and reconstruction
quality against `fixtures/truth.json` (which library code never reads).
