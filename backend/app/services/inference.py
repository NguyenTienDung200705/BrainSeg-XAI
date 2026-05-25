"""
Inference service — tự động detect model type (SMP hay vanilla UNet).
"""
import torch
import numpy as np
from pathlib import Path

from app.models.unet import get_model, UNet
from app.utils.image_utils import (
    preprocess, mask_to_binary,
    apply_colormap_heatmap, overlay_mask, overlay_heatmap, ndarray_to_b64,
)
from app.xai.gradcam import GradCAM, get_target_layer
from app.services.feature_extraction import extract_features
from app.rules.medical_rules import evaluate_risk

WEIGHTS_PATH = Path(__file__).parent.parent.parent / "weights" / "unet_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model      = None
_model_type = "vanilla"
_gradcam    = None

def convert_numpy(obj):
    import numpy as np

    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [convert_numpy(v) for v in obj]

    return obj

def _load_model():
    global _model, _model_type, _gradcam
    if _model is not None:
        return

    if not WEIGHTS_PATH.exists():
        print("[WARN] Weights not found — demo mode (random weights).")
        model, mtype = get_model(pretrained=False)
        model.to(DEVICE).eval()
        _model      = model
        _model_type = mtype
        _gradcam    = GradCAM(model, get_target_layer(model, mtype))
        return

    ckpt = torch.load(WEIGHTS_PATH, map_location=DEVICE)

    # ── Đọc metadata lưu trong checkpoint (nếu có) ──
    arch    = ckpt.get("arch",    "unetplusplus") if isinstance(ckpt, dict) else "unetplusplus"
    encoder = ckpt.get("encoder", "efficientnet-b4") if isinstance(ckpt, dict) else "efficientnet-b4"

    # Lấy state_dict
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and any(k.startswith("encoder") or k.startswith("decoder") for k in ckpt.keys()):
        # SMP raw state_dict
        sd = ckpt
        arch, encoder = "unetplusplus", "efficientnet-b4"
    else:
        sd = ckpt

    # Detect vanilla UNet vs SMP bằng key names
    keys = list(sd.keys()) if isinstance(sd, dict) else []
    is_vanilla = any(k.startswith("inc.") or k.startswith("down") or k.startswith("up") for k in keys)

    if is_vanilla:
        model    = UNet(n_channels=3, n_classes=1)
        mtype    = "vanilla"
        print(f"[Model] Load vanilla UNet từ {WEIGHTS_PATH}")
    else:
        try:
            model, mtype = get_model(arch=arch, encoder=encoder, pretrained=False)
            print(f"[Model] Load SMP {arch}/{encoder} từ {WEIGHTS_PATH}")
        except Exception as e:
            print(f"[Model] SMP fail ({e}), thử vanilla UNet.")
            model = UNet(n_channels=3, n_classes=1)
            mtype = "vanilla"

    model.load_state_dict(sd, strict=False)
    model.to(DEVICE).eval()

    _model      = model
    _model_type = mtype
    _gradcam    = GradCAM(model, get_target_layer(model, mtype))
    print(f"[Model] Ready — type={mtype}, device={DEVICE}")


def predict(img_rgb: np.ndarray) -> dict:
    _load_model()

    tensor, resized = preprocess(img_rgb)
    tensor = tensor.to(DEVICE)

    with torch.no_grad():
        logits = _model(tensor)

    mask = mask_to_binary(logits)

    # Grad-CAM
    try:
        cam = _gradcam.generate(tensor.clone())
    except Exception as e:
        print(f"[GradCAM] Error: {e}")
        cam = np.zeros(resized.shape[:2], dtype=np.float32)

    heatmap_rgb  = apply_colormap_heatmap(cam)
    mask_overlay = overlay_mask(resized, mask)
    cam_overlay  = overlay_heatmap(resized, heatmap_rgb)

    mask_vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
    mask_vis[mask == 1] = [0, 255, 136]

    features = extract_features(mask, mask.shape)
    risk     = evaluate_risk(features)

    result = {
        "original_b64":    ndarray_to_b64(resized),
        "mask_b64":        ndarray_to_b64(mask_vis),
        "overlay_b64":     ndarray_to_b64(mask_overlay),
        "heatmap_b64":     ndarray_to_b64(heatmap_rgb),
        "cam_overlay_b64": ndarray_to_b64(cam_overlay),
        "features": features,
        "risk": {
            "risk_level":      risk.risk_level,
            "risk_score":      risk.risk_score,
            "severity":        risk.severity,
            "risk_color":      risk.risk_color,
            "fired_rules":     risk.fired_rules,
            "recommendations": risk.recommendations,
            "explanation":     risk.explanation,
        },
    }

    return convert_numpy(result)
