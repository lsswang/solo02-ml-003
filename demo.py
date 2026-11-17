"""Runnable examples.

    python demo.py               # legacy independent-pinhole scene
    python demo.py shared_clean  # shared-lens joint reconstruction
"""
import json
import sys
from pathlib import Path

import numpy as np

from bundle import Layout, analyze_layout, evaluate_scene, fit, residuals, solve

FIXTURES = Path(__file__).parent / "fixtures"


def summarize(result):
    keys = ("status", "initial_cost", "final_cost", "layout_rank",
            "free_parameter_count", "pixel_rmse", "all_visible")
    return {k: result.get(k) for k in keys}


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "ordinary"
    scene = json.loads((FIXTURES / (name + ".json")).read_text())

    if name == "ordinary":
        # Legacy entry point: independent pinhole calibrations, fixed poses.
        layout = Layout(scene)
        print(json.dumps({
            "initial_residual_norm": float(np.linalg.norm(residuals(layout, layout.x0))),
        }, indent=2))
        print(json.dumps(summarize(solve(scene)), indent=2))
    else:
        report = analyze_layout(scene)
        result = fit(scene)
        # Re-evaluate the exported scene independently to prove consistency.
        independent = evaluate_scene(result["scene"], result["delta"])
        print(json.dumps({
            "layout": {
                "free_parameter_count": report["free_parameter_count"],
                "layout_rank": report["layout_rank"],
                "missing_dimensions": report["missing_dimensions"],
            },
            "fit": summarize(result),
            "exported_cost_matches": abs(independent["cost"] - result["final_cost"]) < 1e-9,
            "first_observation": result["diagnostics"][0],
        }, indent=2))
