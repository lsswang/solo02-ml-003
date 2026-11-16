import json
from pathlib import Path
from bundle import Layout, residuals, solve

if __name__ == "__main__":
    scene = json.loads((Path(__file__).parent / "fixtures" / "ordinary.json").read_text())
    layout = Layout(scene)
    print(json.dumps({"initial_residual_norm": float((residuals(layout, layout.x0) ** 2).sum() ** .5), "fit": solve(scene)}, indent=2))
