import numpy as np
from scipy import ndimage
from skimage import measure


def extract_features(mask: np.ndarray, img_shape: tuple) -> dict:
    """
    mask: binary H×W uint8
    img_shape: (H, W)
    Returns dict of medical features.
    """
    H, W = img_shape
    total_pixels = H * W

    labeled, num_tumors = ndimage.label(mask)
    tumor_pixels = int(mask.sum())
    tumor_area = float(tumor_pixels)
    occupancy_ratio = round(tumor_pixels / total_pixels * 100, 2)

    if tumor_pixels == 0:
        return {
            "tumor_detected": False,
            "tumor_area_px": 0,
            "tumor_area_cm2": 0.0,
            "occupancy_ratio": 0.0,
            "num_regions": 0,
            "shape_irregularity": 0.0,
            "compactness": 0.0,
            "boundary_complexity": 0.0,
            "midline_shift": False,
            "location": "Không phát hiện khối u",
            "centroid_x": None,
            "centroid_y": None,
        }

    # Convert px to approx cm² (MRI slice ~ 24cm × 24cm for 256×256 resolution)
    px_to_cm = (24.0 / H) ** 2
    tumor_area_cm2 = round(tumor_pixels * px_to_cm, 2)

    # Shape features via regionprops on largest region
    props_list = measure.regionprops(labeled)
    props = max(props_list, key=lambda p: p.area)

    perimeter = props.perimeter if props.perimeter > 0 else 1
    area_p = props.area
    compactness = round((4 * np.pi * area_p) / (perimeter ** 2), 4)
    shape_irregularity = round(1.0 - compactness, 4)

    # Boundary complexity (perimeter / sqrt(area))
    boundary_complexity = round(perimeter / (np.sqrt(area_p) + 1e-6), 4)

    # Centroid & location
    cy, cx = props.centroid
    cx_norm = cx / W
    cy_norm = cy / H

    if cx_norm < 0.45:
        h_loc = "bên trái"
    elif cx_norm > 0.55:
        h_loc = "bên phải"
    else:
        h_loc = "trung tâm"

    if cy_norm < 0.33:
        v_loc = "thùy trán"
    elif cy_norm < 0.66:
        v_loc = "thùy đỉnh/thái dương"
    else:
        v_loc = "thùy chẩm/tiểu não"

    location = f"{v_loc}, {h_loc}"

    # Midline shift: centroid deviates >10% from center
    midline_shift = abs(cx_norm - 0.5) > 0.10

    return {
        "tumor_detected": True,
        "tumor_area_px": tumor_pixels,
        "tumor_area_cm2": tumor_area_cm2,
        "occupancy_ratio": occupancy_ratio,
        "num_regions": num_tumors,
        "shape_irregularity": shape_irregularity,
        "compactness": compactness,
        "boundary_complexity": boundary_complexity,
        "midline_shift": midline_shift,
        "location": location,
        "centroid_x": round(float(cx), 1),
        "centroid_y": round(float(cy), 1),
    }
