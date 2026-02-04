## Copyright & Usage

© 2026 Divyanshu Kumar. All rights reserved.

This repository and its contents are shared for **viewing and evaluation purposes only** (e.g., capstone review, learning reference).

**No permission is granted** to use, copy, modify, distribute, publish, or deploy any part of this project (code, UI, documentation, or assets) in any form **without explicit written permission** from the author.

If you would like to use or build upon this work, please contact me and include proper attribution:
**“Original work by Divyanshu Kumar — GitHub: https://github.com/kdivyanshu5”**

# Project README
# VEHICLE-INTEGRATED-UAV-SURVEILLANCE-SYSTEM
Unmanned Aerial Vehicles (UAVs) pose security and surveillance challenges, especially in restricted and sensitive zones. Manual monitoring is inefficient and error-prone. Detect UAVs in real time using vision-based techniques.->Integrate surveillance with vehicle-mounted systems.->Provide fast and accurate alerts for security response.

## Overview
This repository contains a YOLO-based drone detection app (desktop GUI) and a Flask-based web gallery to view saved detections.

- `app.py` — desktop application (Tkinter) that runs the live detector and saves detection snapshots.
- `webapp.py` — Flask backend that serves a gallery of saved detection images.

> Credentials used by both the GUI and web backend: **username:** `admin` — **password:** `password123`

---

## Python and virtual environment (Python 3.10)
1. Install Python 3.10 on your system (make sure `python3.10` is available in your PATH).
2. Create a venv inside the project:

```bash
python3.10 -m venv .venv
```

3. Activate the venv:

- macOS / Linux

```bash
source .venv/bin/activate
```

- Windows (PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
```

- Windows (cmd)

```cmd
.\.venv\Scripts\activate.bat
```

4. Upgrade pip and setuptools inside the activated venv:

```bash
python -m pip install --upgrade pip setuptools wheel
```

---

## Recommended Python packages
Below are the packages this project uses. Some packages (PyTorch and related) need a CUDA-enabled build if you want GPU acceleration.

### Core (Flask backend + utilities)
```
pip install Flask
pip install numpy Pillow
pip install opencv-python-headless   # headless for server environments
pip install pyserial
pip install ultralytics
```

### GPU / CUDA (example)
PyTorch must match your CUDA toolkit on the machine (the command below is an example — pick the wheel that matches your CUDA version):

```bash
# Example for CUDA-enabled pip wheels (change cu117/cu118/cu121 to match your system)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

- If you don't have a CUDA GPU, install the CPU-only build of torch or omit the CUDA index URL.

### Optional / dev
```
pip install jupyterlab   # optional
pip install gunicorn     # if you want to serve Flask in production
```

---

## Quick start — run the app (two terminals)
You should open two terminals (activate the same `.venv` in both):

**Terminal 1 — run the desktop detector (GUI):**

```bash
# inside project and with .venv activated
python app.py
```

**Terminal 2 — run the Flask gallery:**

```bash
# inside project and with .venv activated
python webapp.py
```

This keeps the detector UI (camera + model) running in one terminal while the web gallery serves the saved images in the other.

---

## Important notes
- `app.py` and `webapp.py` both expect a `SAVE_DIR` path where detection images are stored. Make sure that path exists and is writable.
- The desktop app uses `ultralytics` + `torch` for YOLO inference — if you want GPU acceleration, install the CUDA-enabled PyTorch wheel that matches your system's CUDA drivers.
- `opencv-python-headless` is recommended for server/headless environments. If you need GUI features locally, replace with `opencv-python`.

---

## Creating `requirements.txt`
After installing packages inside the venv, pin them with:

```bash
pip freeze > requirements.txt
```

---

## Files of interest
- `app.py` — desktop application logic
- `webapp.py` — Flask backend and gallery
- `templates/` — HTML templates used by Flask (login, gallery)


---

