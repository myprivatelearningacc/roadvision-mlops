# RoadVision AI — Phase 6 Starter

This zip is an *overlay* for the Git/DVC repository created in Phase 5. It does not include the trained model, datasets, notebooks, or `.git/` history.

## 1. Restore existing Git repository from Google Drive bundle

Download `MyDrive/RoadVision/git_backup/roadvision-mlops.bundle` to your computer. In PowerShell:

```powershell
cd C:\Projects
git clone "$env:USERPROFILE\Downloads\roadvision-mlops.bundle" roadvision-mlops
cd roadvision-mlops
git log --oneline -3
```

## 2. Add Phase 6 files

Extract this zip elsewhere, copy `backend/`, `frontend/`, `docker-compose.yml`, `.dockerignore`, `README_PHASE6.md` into the cloned `roadvision-mlops` folder. Do NOT overwrite `.git`, `.dvc`, `dvc.yaml`, `dvc.lock`, `src/verify_data.py`.

Copy notebook files from Drive/Colab into `notebooks/` (e.g. `RoadVisionv1.ipynb`, `RoadVisionv2.ipynb`, `RoadVisionv3.ipynb`). Notebook output may contain large embedded images. Consider clearing output before Git commit, or store only links.

Download `MyDrive/RoadVision/models/v2/best.pt` to `models/yolo11n_v2.pt`. It should match the DVC-tracked checkpoint, not a new retraining artifact. The DVC remote created on Colab is a Colab-mounted path and won't work directly in Windows; later configure another reachable remote, or use matching downloaded checkpoints now.

Add lines from `.gitignore.phase6` to `.gitignore` without overwriting DVC-added rules. Do not add model binaries or data ZIPs to Git.

## 3. Backend on Windows

Prefer Python 3.11 or 3.12 on your **personal** machine (CPU inference is fine). From the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
python -m uvicorn backend.app.main:app --reload --port 8000
```

If `py -3.11` is unavailable, use a locally installed compatible Python interpreter. Avoid installing packages into your system Python. Visit `http://127.0.0.1:8000/docs` and `http://127.0.0.1:8000/health`.

## 4. React frontend

Requires compatible Node.js/npm. In another PowerShell terminal:

```powershell
cd frontend
npm install
npm run dev
```

Visit `http://127.0.0.1:5173`. Upload a JPEG/PNG/WebP and press **Run detection**.

## 5. Docker backend (optional)

From repository root: `docker compose up --build`. The frontend still runs with `npm run dev` locally.

## 6. Git

```powershell
git add backend frontend docker-compose.yml .dockerignore README_PHASE6.md .gitignore notebooks
git commit -m "Add FastAPI inference and React demo"
```

The starter includes basic validation, logging of latency and detection count (no saved images), and an explicit confidence slider. It is NOT production hardened (no auth, quotas, rate limiter, persistent inference monitoring, automated DVC pull, or CI/CD yet).
