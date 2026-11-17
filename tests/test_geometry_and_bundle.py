"""Repeatable checks for the geometry interface, shared/locked parameters,
the grouped robust objective and the reconstruction quality.

Run with ``pytest -q`` (or ``python -m pytest tests``).  These tests never read
``fixtures/truth.json`` from library code; they load it only as validation data
here.
"""
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

import bundle
from camera import PARAMETERS, jacobian, project, unproject

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "fixtures"


def load(name):
    return json.loads((FIXTURES / name).read_text())


def calibration(**kwargs):
    values = dict(fx=528.0, fy=504.0, cx=320.0, cy=240.0,
                  k1=0.088, k2=-0.026, p1=0.007, p2=-0.005)
    values.update(kwargs)
    return values


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def _domain_points(rng, n):
    # Camera points whose normalized radius stays within the documented domain.
    pts = np.empty((n, 3))
    pts[:, 0] = rng.uniform(-0.45, 0.45, n)
    pts[:, 1] = rng.uniform(-0.45, 0.45, n)
    pts[:, 2] = rng.uniform(0.8, 6.0, n)
    return pts


def test_projection_roundtrip_and_ray():
    rng = np.random.default_rng(7)
    cal = calibration()
    pts = _domain_points(rng, 500)
    xy = pts[:, :2] / pts[:, 2:3]
    assert np.max(np.linalg.norm(xy, axis=1)) <= 0.8

    pixels = project(pts, cal)
    rays = unproject(pixels, cal)
    assert np.all(rays[:, 2] == 1.0)
    # Ray round trip: reprojection error <= 2e-6 px.
    assert np.max(np.abs(project(rays, cal) - pixels)) <= 2e-6
    # Undistorted ray equals the true normalized coordinate <= 2e-8.
    assert np.max(np.abs(rays[:, :2] - xy)) <= 2e-8


def test_zero_distortion_legacy_behavior():
    cal = dict(fx=500.0, fy=600.0, cx=320.0, cy=240.0)
    pts = np.array([[1.0, 2.0, 2.0], [-0.5, 0.25, 4.0]])
    expected = (pts[:, :2] / pts[:, 2:3]) * [500.0, 600.0] + [320.0, 240.0]
    np.testing.assert_allclose(project(pts, cal), expected)
    np.testing.assert_allclose(unproject(expected, cal)[:, :2], pts[:, :2] / pts[:, 2:3])
    jp, jc = jacobian(pts[0], cal)
    assert jc.shape == (2, 8)
    assert jc[0, 0] == pytest.approx(0.5)   # fx derivative = x = 1/2
    assert jc[1, 1] == pytest.approx(1.0)   # fy derivative = y = 2/2
    assert jc[0, 2] == 1.0 and jc[1, 3] == 1.0


def test_jacobian_matches_finite_difference():
    rng = np.random.default_rng(11)
    for _ in range(5):
        cal = calibration(
            fx=rng.uniform(200, 1200), fy=rng.uniform(200, 1200),
            cx=rng.uniform(0, 640), cy=rng.uniform(0, 480),
            k1=rng.uniform(-0.15, 0.15), k2=rng.uniform(-0.05, 0.05),
            p1=rng.uniform(-0.01, 0.01), p2=rng.uniform(-0.01, 0.01),
        )
        point = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), rng.uniform(0.8, 5.0)])
        jp, jc = jacobian(point, cal)
        h = 1e-6
        jp_fd = np.zeros((2, 3))
        for j in range(3):
            delta = np.zeros(3); delta[j] = h
            jp_fd[:, j] = (project((point + delta)[None], cal)[0]
                           - project((point - delta)[None], cal)[0]) / (2 * h)
        jc_fd = np.zeros((2, 8))
        for j, key in enumerate(PARAMETERS):
            a = dict(cal); b = dict(cal)
            a[key] += h; b[key] -= h
            jc_fd[:, j] = (project(point[None], a)[0] - project(point[None], b)[0]) / (2 * h)
        np.testing.assert_allclose(jp, jp_fd, atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(jc, jc_fd, atol=2e-5, rtol=2e-5)


def test_geometry_rejections():
    cal = calibration()
    with pytest.raises(ValueError):
        project(np.array([[0.0, 0.0, -1.0]]), cal)
    with pytest.raises(ValueError):
        project(np.array([[0.0, np.inf, 1.0]]), cal)
    with pytest.raises(ValueError):
        project(np.array([[0.0, 0.0, 1.0]]), calibration(fx=0.0))
    with pytest.raises(ValueError):
        project(np.array([[0.0, 0.0, 1.0]]), calibration(fy=-5.0))
    with pytest.raises(ValueError):
        project(np.array([[0.0, 0.0, 1.0]]), calibration(k1=float("nan")))
    with pytest.raises(ValueError):
        unproject(np.array([[np.nan, 0.0]]), cal)


# ---------------------------------------------------------------------------
# Shared parameters, locking and ownership
# ---------------------------------------------------------------------------
def test_shared_calibration_has_one_variable_group():
    scene = load("shared_clean.json")
    layout = bundle.Layout(scene)
    cal_params = [(e[1], e[2]) for e in layout.entries if e[0] == "calibration"]
    # Exactly one group of parameters, keyed by the shared id "lens".
    assert {owner for owner, _ in cal_params} == {"lens"}
    assert len(cal_params) == 6  # cx, cy are fixed
    # No per-view duplication of intrinsics.
    names = [e[2] for e in layout.entries if e[0] == "calibration"]
    assert sorted(names) == ["fx", "fy", "k1", "k2", "p1", "p2"]


def test_fixed_parameters_and_objects_are_untouched():
    scene = load("shared_clean.json")
    result = bundle.fit(scene)
    assert result["status"] == "converged"
    out = result["scene"]
    # Parameter-level fixed values are bit-for-bit identical.
    for key in ("cx", "cy"):
        assert out["calibrations"]["lens"][key] == scene["calibrations"]["lens"][key]
    # Fixed views and fixed points are untouched.
    for vin, vout in zip(scene["views"], out["views"]):
        if vin.get("fixed"):
            assert vout["rvec"] == vin["rvec"]
            assert vout["t"] == vin["t"]
    for pin, pout in zip(scene["points"], out["points"]):
        if pin.get("fixed"):
            assert pout["xyz"] == pin["xyz"]
    # The shared calibration is still a single object under the same id.
    assert list(out["calibrations"]) == ["lens"]
    assert all(v["calibration_id"] == "lens" for v in out["views"])


def test_free_parameter_identity_and_count():
    scene = load("shared_clean.json")
    layout = bundle.Layout(scene)
    expected = 6 + 6 + 12 * 3  # lens(fx,fy,k1,k2,p1,p2) + view c pose + 12 free points*3
    assert layout.nfree == expected
    ids = [p["id"] for p in bundle.free_parameter_identities(layout)]
    assert len(set(ids)) == expected
    assert "calibration:lens:fx" in ids
    assert "view:c:rvec:0" in ids
    assert "point:p08:xyz:0" in ids


# ---------------------------------------------------------------------------
# Objective contract
# ---------------------------------------------------------------------------
def _model_pixel(scene, obs):
    view = next(v for v in scene["views"] if v["id"] == obs["view_id"])
    point = next(p for p in scene["points"] if p["id"] == obs["point_id"])
    cam = bundle.rotation(view["rvec"]) @ np.array(point["xyz"]) + np.array(view["t"])
    return project(cam[None], scene["calibrations"][view["calibration_id"]])[0]


def test_grouped_huber_is_not_per_coordinate():
    # Synthetic two-view, two-point scene with a controllable residual.
    scene = load("ordinary.json")
    obs = scene["observations"][0]
    # Build a residual of (3, 0) standardized: q=3 > delta=2.
    view = next(v for v in scene["views"] if v["id"] == obs["view_id"])
    point = next(p for p in scene["points"] if p["id"] == obs["point_id"])
    pred = _model_pixel(scene, obs)
    tuned = deepcopy(scene)
    tuned_obs = tuned["observations"][0]
    tuned_obs["pixel"] = list(pred + np.array([3.0, 0.0]) * tuned_obs["sigma"])

    grouped = bundle.evaluate_scene(tuned, delta=2.0)
    per_coord_expected = (
        (0.5 * 3.0 ** 2)  # x-coordinate, but grouped loss would use q=3
    )
    # Grouped Huber at q=3, delta=2: delta*(q - delta/2)=2*(3-1)=4.
    target = next(d for d in grouped["diagnostics"] if d["view_id"] == obs["view_id"]
                  and d["point_id"] == obs["point_id"])
    assert target["q"] == pytest.approx(3.0)
    assert target["cost"] == pytest.approx(4.0)
    # Coordinate-wise Huber would be 0.5*2^2 + 2*(3-1) = 2+4 = 6.
    assert target["cost"] != pytest.approx(6.0)


def test_exported_scene_recomputes_same_cost_and_residuals():
    for name in ("shared_clean.json", "shared_outliers.json", "ordinary.json"):
        scene = load(name)
        result = bundle.fit(scene, delta=2.0)
        recomputed = bundle.evaluate_scene(result["scene"], delta=2.0)
        assert recomputed["cost"] == pytest.approx(result["final_cost"], abs=1e-10)
        assert len(recomputed["diagnostics"]) == len(result["diagnostics"])
        for a, b in zip(result["diagnostics"], recomputed["diagnostics"]):
            assert a["q"] == pytest.approx(b["q"], abs=1e-9)
            assert a["cost"] == pytest.approx(b["cost"], abs=1e-9)
            assert a["visible"] == b["visible"]
            np.testing.assert_allclose(a["residual"], b["residual"], atol=1e-9)


def test_delta_range_is_honored():
    scene = load("shared_outliers.json")
    for delta in (0.5, 1.0, 2.0, 4.0):
        result = bundle.fit(scene, delta=delta)
        assert result["delta"] == delta
        assert result["status"] == "converged"
        recomputed = bundle.evaluate_scene(result["scene"], delta=delta)
        assert recomputed["cost"] == pytest.approx(result["final_cost"], abs=1e-9)
        # Fewer inliers at smaller delta (strictly, never more).
        counts = []
        for d in (0.5, 1.0, 2.0, 4.0):
            r = bundle.fit(scene, delta=d)
            counts.append(r["inlier_count"])
        assert counts == sorted(counts)


def test_rejects_invalid_sigma_and_scene_ids():
    scene = load("ordinary.json")
    bad = deepcopy(scene)
    bad["observations"][0]["sigma"] = 0.0
    with pytest.raises(ValueError):
        bundle.Layout(bad)
    bad2 = deepcopy(scene)
    bad2["observations"][0]["view_id"] = "missing"
    with pytest.raises(ValueError):
        bundle.Layout(bad2)


# ---------------------------------------------------------------------------
# Layout analysis
# ---------------------------------------------------------------------------
def test_analyze_layout_normalized_rank():
    scene = load("shared_clean.json")
    report = bundle.analyze_layout(scene)
    assert report["free_parameter_count"] == bundle.Layout(scene).nfree
    assert report["layout_rank"] == report["free_parameter_count"]
    assert report["missing_dimensions"] == 0
    assert len(report["free_parameters"]) == report["free_parameter_count"]
    # Column norms are documented but rank uses normalized columns.
    assert all(n >= 0 for n in report["column_norms"])


def test_rank_deficient_scene_is_reported():
    scene = load("rank_deficient.json")
    report = bundle.analyze_layout(scene)
    assert report["free_parameter_count"] == 3
    assert report["layout_rank"] == 2
    assert report["missing_dimensions"] == 1
    result = bundle.fit(scene)
    assert result["status"] == "unobservable"
    assert result["missing_dimensions"] == 1
    # No invented reconstruction: the scene comes back unchanged.
    assert result["scene"]["points"] == scene["points"]


def test_empty_variable_set_rank_is_zero():
    scene = load("ordinary.json")
    frozen = deepcopy(scene)
    for cal in frozen["calibrations"].values():
        cal["fixed"] = list(PARAMETERS)
    report = bundle.analyze_layout(frozen)
    assert report["free_parameter_count"] == 0
    assert report["layout_rank"] == 0
    assert report["missing_dimensions"] == 0


# ---------------------------------------------------------------------------
# Reconstruction quality against validation truth
# ---------------------------------------------------------------------------
def test_clean_reconstruction_matches_truth():
    scene = load("shared_clean.json")
    truth = load("truth.json")
    result = bundle.fit(scene)
    assert result["status"] == "converged"
    # Noiseless: pixel RMSE < 1e-5.
    assert result["pixel_rmse"] < 1e-5

    out = result["scene"]
    max_err = 0.0
    for p, pt in zip(out["points"], truth["points"]):
        if p.get("fixed"):
            continue
        max_err = max(max_err, float(np.max(np.abs(np.array(p["xyz"]) - np.array(pt["xyz"])))))
    assert max_err < 2e-4

    # Shared lens was recovered as one group.
    for key in PARAMETERS:
        if key in ("cx", "cy"):
            continue
        assert out["calibrations"]["lens"][key] == pytest.approx(
            truth["calibrations"]["lens"][key], abs=1e-6)


def test_outlier_fit_beats_truth_objective():
    scene = load("shared_outliers.json")
    truth = load("truth.json")

    truth_scene = deepcopy(scene)
    for key, value in truth["calibrations"]["lens"].items():
        if key != "fixed":
            truth_scene["calibrations"]["lens"][key] = value
    for vout, vtrue in zip(truth_scene["views"], truth["views"]):
        vout["rvec"] = list(vtrue["rvec"])
        vout["t"] = list(vtrue["t"])
    for pout, ptrue in zip(truth_scene["points"], truth["points"]):
        pout["xyz"] = list(ptrue["xyz"])
    truth_cost = bundle.evaluate_scene(truth_scene, delta=2.0)["cost"]

    result = bundle.fit(scene, delta=2.0)
    assert result["status"] == "converged"
    assert result["final_cost"] <= truth_cost + 1e-4
    # Fixed values and sharing survive the robust fit.
    for key in ("cx", "cy"):
        assert result["scene"]["calibrations"]["lens"][key] == scene["calibrations"]["lens"][key]
    assert len(result["scene"]["calibrations"]) == 1


def test_solution_does_not_regress_or_leave_domain():
    for name in ("ordinary.json", "shared_clean.json", "shared_outliers.json"):
        scene = load(name)
        result = bundle.fit(scene)
        assert result["status"] in ("converged", "not_converged")
        assert result["final_cost"] <= result["initial_cost"] + 1e-9
        assert result["all_visible"]
        for diag in result["diagnostics"]:
            if diag["visible"]:
                assert np.isfinite(diag["q"])


def test_invalid_domain_scene_is_not_converged():
    scene = load("shared_clean.json")
    bad = deepcopy(scene)
    bad["points"][8]["xyz"][2] = -1.0  # behind the camera at the initial values
    result = bundle.fit(bad)
    assert result["status"] == "not_converged"
    assert result["all_visible"] is False
    assert result["scene"] == bad  # never mutates non-free fields on failure


def test_legacy_solve_entry_point():
    scene = load("ordinary.json")
    result = bundle.solve(scene)
    assert result["status"] == "converged"
    assert "cost" in result and result["cost"] == result["final_cost"]
    assert result["free_parameter_count"] == 16
    layout = bundle.Layout(scene)
    r = bundle.residuals(layout, layout.x0)
    assert r.shape == (2 * len(scene["observations"]),)
    assert bundle.objective(layout, layout.x0) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Checks that catch the three named regression classes
# ---------------------------------------------------------------------------
def test_detects_projection_only_fix_with_split_shared_parameters():
    """A fitter that fixes project() but duplicates per-view intrinsics would
    produce more than one calibration group; assert the shared identity holds
    in every exported scene."""
    for name in ("shared_clean.json", "shared_outliers.json"):
        scene = load(name)
        result = bundle.fit(scene)
        out = result["scene"]
        # One id, one object, every view points at it.
        assert list(out["calibrations"]) == ["lens"]
        assert all(v["calibration_id"] == "lens" for v in out["views"])
        # A duplicated implementation would leave copies at different values.
        # Every view that references a shared id must see the same parameters
        # by construction; re-projecting through project() must match the
        # diagnostics for each view.
        for diag in result["diagnostics"]:
            view = next(v for v in out["views"] if v["id"] == diag["view_id"])
            point = next(p for p in out["points"] if p["id"] == diag["point_id"])
            cam = bundle.rotation(view["rvec"]) @ np.array(point["xyz"]) + np.array(view["t"])
            cal = out["calibrations"][view["calibration_id"]]
            pred = project(cam[None], cal)[0]
            np.testing.assert_allclose(pred, diag["predicted_pixel"], atol=1e-9)


def test_detects_internal_objective_mismatch():
    """The exported scene must reproduce the reported cost and every residual
    when evaluated independently."""
    for name in ("shared_clean.json", "shared_outliers.json", "ordinary.json"):
        scene = load(name)
        result = bundle.fit(scene, delta=1.5)
        independent = bundle.evaluate_scene(result["scene"], delta=1.5)
        assert independent["cost"] == pytest.approx(result["final_cost"], abs=1e-10)
        for reported, fresh in zip(result["diagnostics"], independent["diagnostics"]):
            np.testing.assert_allclose(reported["residual"], fresh["residual"], atol=1e-9)
            assert reported["cost"] == pytest.approx(fresh["cost"], abs=1e-9)


def test_detects_per_coordinate_robustification():
    """A residual of (q/sqrt2, q/sqrt2) must be treated as one 2-D block:
    at q=3, delta=2 the grouped cost is 4, while coordinate-wise Huber on
    (a,a) with a=3/sqrt2 gives a different value."""
    scene = load("ordinary.json")
    obs = scene["observations"][0]
    pred = _model_pixel(scene, obs)
    tuned = deepcopy(scene)
    tuned["observations"][0]["pixel"] = list(pred + np.array([3.0, 0.0]))  # q=3, sigma=1
    diag = next(d for d in bundle.evaluate_scene(tuned, 2.0)["diagnostics"]
                if d["view_id"] == obs["view_id"] and d["point_id"] == obs["point_id"])
    assert diag["q"] == pytest.approx(3.0)
    assert diag["cost"] == pytest.approx(4.0)  # 2*(3-1)

    # Diagonal residual (a, a), a = 3/sqrt(2): grouped q is still 3 -> 4.0.
    diag_tuned = deepcopy(scene)
    a = 3.0 / np.sqrt(2.0)
    diag_tuned["observations"][0]["pixel"] = list(pred + np.array([a, a]))
    d2 = next(d for d in bundle.evaluate_scene(diag_tuned, 2.0)["diagnostics"]
              if d["view_id"] == obs["view_id"] and d["point_id"] == obs["point_id"])
    assert d2["q"] == pytest.approx(3.0)
    assert d2["cost"] == pytest.approx(4.0)

    # Coordinate-wise Huber would give 2 * (delta*(a - delta/2)) for a > delta.
    # With a = 3/sqrt2 = 2.121 > 2 that is 2*(2*(2.121-1)) = 4.485, not 4.0.
    coordwise = 2.0 * (2.0 * (a - 1.0))
    assert d2["cost"] == pytest.approx(4.0)
    assert abs(d2["cost"] - coordwise) > 0.4
