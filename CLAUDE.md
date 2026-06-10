# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BrainSeg-XAI (BrainAI) is an explainable medical AI system for brain tumor segmentation from MRI images. It uses a U-Net / U-Net++ model to produce binary tumor masks, extracts morphological features, applies a rule-based medical risk engine, and visualizes results via Grad-CAM heatmaps on a React frontend.

## Commands

### Backend (FastAPI)

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs: http://localhost:8000/docs

### Frontend (React)

```bash
cd frontend
npm install --legacy-peer-deps
npm start         # dev server at http://localhost:3000
npm run build     # production build
```

### Quick Start (both services)

```bash
./start.sh        # auto-detects Docker or local mode
```

### Docker (production)

```bash
cp your_model.pth backend/weights/unet_best.pth
docker-compose up --build
```

## Architecture

The request pipeline flows through these layers in order:

```
POST /api/v1/predict (multipart image)
  → api/predict.py          endpoint validation (type, size ≤ 20MB)
  → utils/image_utils.py    load bytes → RGB ndarray, resize to 256×256, ImageNet normalize
  → services/inference.py   lazy-load model, run forward pass (logits → sigmoid → binary mask)
  → xai/gradcam.py          Grad-CAM heatmap on target layer (hooks registered at load time)
  → services/feature_extraction.py   morphological features from mask (area, shape, boundary, midline)
  → rules/medical_rules.py  rule engine → RuleResult with score 0-100 and risk tier
  ← JSON: 5 base64 images + features dict + risk dict
```

### Model loading (`services/inference.py`)

Models are loaded lazily on first request. The loader auto-detects checkpoint type by inspecting `state_dict` keys:
- Keys starting with `inc.` / `down` / `up` → vanilla `UNet` (`models/unet.py`)
- Otherwise → SMP `UNet++` with `efficientnet-b4` encoder via `segmentation_models_pytorch`

If `backend/weights/unet_best.pth` is absent, the system starts in **demo mode** with random weights (all features still work, predictions are meaningless).

### Model variants (`models/unet.py`)

Two model families are supported:
- **Vanilla UNet**: `UNet(n_channels=3, n_classes=1)` — pure PyTorch, backward-compatible with old checkpoints
- **SMP models**: `build_smp_model(arch, encoder)` via `segmentation-models-pytorch` — supports `unetplusplus`, `unet`, `deeplabv3plus`, `fpn`, `manet` with various encoders

### Grad-CAM (`xai/gradcam.py`)

`GradCAM` registers forward/backward hooks at construction time on a `target_layer`. The target layer differs by model type — `get_target_layer()` returns `model.up4.conv.conv[-3]` for vanilla UNet and `model.segmentation_head[0]` for SMP models.

### Medical Rule Engine (`rules/medical_rules.py`)

A pure rule-based scoring system (no ML). Six independent rules fire on extracted features and add to a score capped at 100. Risk tiers: Low (0–29), Medium (30–49), High (50–69), Very High (70+). The `RuleResult` dataclass holds score, tier, color, fired rules, and recommendations.

### Feature Extraction (`services/feature_extraction.py`)

Computes morphological features from the binary mask using `scipy.ndimage` (connected components) and `skimage.measure.regionprops` (perimeter, centroid, area). Key features: `occupancy_ratio`, `shape_irregularity` (1 - compactness), `boundary_complexity` (perimeter/√area), `midline_shift` (centroid deviates >10% from horizontal center), `num_regions`.

### Frontend (`frontend/src/`)

- `App.js` — orchestrates upload → API call → state distribution to panels
- `services/api.js` — Axios calls to backend; dev proxy configured in `package.json` (`http://localhost:8000`)
- `components/ImageViewer.jsx` — 5-view image explorer showing original, mask, overlay, heatmap, cam_overlay (all received as base64 strings)
- `components/RiskPanel.jsx` — risk gauge + fired rules + recommendations
- `components/NeuralCanvas.jsx` — animated background (purely decorative)

### Image normalization

All images are resized to **256×256** with ImageNet normalization (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`). The px→cm² conversion assumes a 24 cm × 24 cm MRI slice at 256×256 resolution.

## Training

Notebooks are in `notebooks/`:
- `training.ipynb` — primary training notebook (LGG MRI Segmentation dataset)
- `training1.ipynb` — variant
- `quick_retrain.py` — script form for quick retraining

Dataset: [LGG MRI Segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) — 3,929 MRI brain images with ground truth masks.

Save checkpoints with arch/encoder metadata so the inference loader can reconstruct the correct model:
```python
torch.save({"arch": "unetplusplus", "encoder": "efficientnet-b4", "model_state_dict": model.state_dict()}, "unet_best.pth")
```
