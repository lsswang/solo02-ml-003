# Small multiview geometry starter

Python 3.11; install requirements.txt. Run `python demo.py` for the existing independent-pinhole-calibration example.

`camera.py` is a directly callable geometry module. `bundle.py` currently fits independent per-view pinhole calibrations, with entire poses/points optionally fixed. It is a working starting point for the extension specified in the question, not the final multiview feature set.

The scene format is JSON. World-to-camera coordinates are `R(rvec) @ xyz + t`. IDs, missing observations and explicit fixed values are significant. `fixtures/truth.json` is validation data and must not be read by a fitting implementation. `ordinary.json` demonstrates the old interface; the other fixtures exercise the requested extension.

Dependencies may be added. No external images, model downloads, services or GPU are needed.
