import cv2
import numpy as np
from PIL import Image
import io
import base64


TARGET_SIZE = (256, 256)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def load_image_from_bytes(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Không thể đọc ảnh. Vui lòng thử lại với file ảnh hợp lệ.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def preprocess(img_rgb: np.ndarray):
    """Return (tensor NCHW float32, original_rgb resized)"""
    import torch
    resized = cv2.resize(img_rgb, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
    norm = resized.astype(np.float32) / 255.0
    for c in range(3):
        norm[:, :, c] = (norm[:, :, c] - MEAN[c]) / STD[c]
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0)  # 1×3×H×W
    return tensor, resized


def mask_to_binary(logits_tensor) -> np.ndarray:
    import torch
    prob = torch.sigmoid(logits_tensor).squeeze().cpu().numpy()
    return (prob > 0.5).astype(np.uint8)


def ndarray_to_b64(arr: np.ndarray, ext: str = "png") -> str:
    pil = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format=ext.upper())
    return base64.b64encode(buf.getvalue()).decode()


def apply_colormap_heatmap(gray: np.ndarray) -> np.ndarray:
    """gray: H×W float [0,1] → RGB uint8"""
    gray_u8 = (gray * 255).clip(0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(gray_u8, cv2.COLORMAP_JET)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def overlay_mask(img_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    overlay = img_rgb.copy()
    overlay[mask == 1] = [255, 80, 80]
    return cv2.addWeighted(img_rgb, 1 - alpha, overlay, alpha, 0)


def overlay_heatmap(img_rgb: np.ndarray, heatmap_rgb: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    return cv2.addWeighted(img_rgb, 1 - alpha, heatmap_rgb, alpha, 0)
