# BrainSeg-XAI — Hướng dẫn tổng quan dự án

**Hệ thống AI phân đoạn khối u não từ ảnh MRI với khả năng giải thích quyết định (XAI)**

---

## Mục lục

1. [Tổng quan](#1-tổng-quan)
2. [Kiến trúc hệ thống](#2-kiến-trúc-hệ-thống)
3. [Cấu trúc thư mục](#3-cấu-trúc-thư-mục)
4. [Yêu cầu hệ thống](#4-yêu-cầu-hệ-thống)
5. [Hướng dẫn cài đặt](#5-hướng-dẫn-cài-đặt)
6. [Cách sử dụng](#6-cách-sử-dụng)
7. [API Reference](#7-api-reference)
8. [Cấu trúc Response JSON](#8-cấu-trúc-response-json)
9. [Mô hình học sâu](#9-mô-hình-học-sâu)
10. [Huấn luyện mô hình](#10-huấn-luyện-mô-hình)
11. [Giải thích kết quả (XAI)](#11-giải-thích-kết-quả-xai)
12. [Bộ máy luật y khoa](#12-bộ-máy-luật-y-khoa)
13. [Câu hỏi thường gặp](#13-câu-hỏi-thường-gặp)
14. [Tuyên bố từ chối trách nhiệm](#14-tuyên-bố-từ-chối-trách-nhiệm)

---

## 1. Tổng quan

BrainSeg-XAI là hệ thống hỗ trợ chẩn đoán u não kết hợp **bốn thành phần** trong một pipeline:

| Thành phần | Công nghệ | Đầu ra |
|---|---|---|
| Phân đoạn hình ảnh | U-Net / U-Net++ (PyTorch) | Mặt nạ nhị phân 256×256 |
| Giải thích quyết định | Grad-CAM | Bản đồ nhiệt attention |
| Trích xuất đặc trưng | scipy + skimage | 9 chỉ số hình thái học |
| Đánh giá rủi ro | Rule-based engine | Điểm 0–100 + 4 mức rủi ro |

Giao diện web React cung cấp 5 chế độ xem ảnh, bảng đặc trưng, đồng hồ rủi ro và khuyến nghị lâm sàng tự động.

> **Lưu ý quan trọng:** Hệ thống này chỉ dành cho mục đích nghiên cứu và học thuật. Không sử dụng thay thế chẩn đoán y khoa chính thức.

---

## 2. Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────────┐
│                        FRONTEND (React)                         │
│  UploadZone → ProcessingPanel → ImageViewer (5 views)          │
│                             → RiskPanel (gauge + rules)         │
└──────────────────────────────┬──────────────────────────────────┘
                               │ POST /api/v1/predict (multipart)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                       BACKEND (FastAPI)                         │
│                                                                 │
│  api/predict.py  ─────────────────────────────────────────────┐│
│      │ load_image_from_bytes()                                 ││
│      ▼                                                          ││
│  utils/image_utils.py                                          ││
│      resize(256×256) + ImageNet normalize → tensor [1,3,H,W]  ││
│      │                                                          ││
│      ▼                                                          ││
│  services/inference.py                                         ││
│      _load_model() [lazy, auto-detect checkpoint type]         ││
│      model(tensor) → logits → sigmoid → binary mask            ││
│      │                                                          ││
│      ├──▶ xai/gradcam.py                                       ││
│      │        backward(sigmoid.sum()) → attention heatmap      ││
│      │                                                          ││
│      ├──▶ services/feature_extraction.py                       ││
│      │        ndimage.label + regionprops → 9 features         ││
│      │                                                          ││
│      └──▶ rules/medical_rules.py                               ││
│               6 rules + additive scoring → RuleResult          ││
│                                                                 ││
│  Response JSON: 5 base64 images + features + risk              ││
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Cấu trúc thư mục

```
BrainSeg-XAI/
├── backend/
│   ├── app/
│   │   ├── main.py                      # FastAPI app + CORS middleware
│   │   ├── api/
│   │   │   └── predict.py               # POST /predict, GET /health
│   │   ├── models/
│   │   │   └── unet.py                  # UNet, build_smp_model(), get_model()
│   │   ├── services/
│   │   │   ├── inference.py             # Lazy loading, prediction pipeline
│   │   │   └── feature_extraction.py    # Morphological feature computation
│   │   ├── rules/
│   │   │   └── medical_rules.py         # Rule-based risk scoring (RuleResult)
│   │   ├── utils/
│   │   │   └── image_utils.py           # Preprocess, colormap, overlay utils
│   │   └── xai/
│   │       └── gradcam.py               # GradCAM class + get_target_layer()
│   ├── weights/
│   │   └── unet_best.pth                # ← Đặt file model vào đây
│   ├── Dockerfile
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── App.js                       # Root component, state management
│   │   ├── index.css                    # Design tokens + global styles
│   │   ├── components/
│   │   │   ├── Header.jsx               # Logo + API status indicator
│   │   │   ├── UploadZone.jsx           # Drag-and-drop file upload
│   │   │   ├── ProcessingPanel.jsx      # Progress ring + stage list
│   │   │   ├── ImageViewer.jsx          # 5-tab image viewer
│   │   │   ├── RiskPanel.jsx            # SVG gauge + features + rules
│   │   │   └── NeuralCanvas.jsx         # Animated background canvas
│   │   └── services/
│   │       └── api.js                   # Axios client (predictTumor, checkHealth)
│   ├── Dockerfile
│   └── nginx.conf
│
├── notebooks/
│   ├── training.ipynb                   # Primary training notebook (LGG dataset)
│   ├── training1.ipynb                  # Variant training notebook
│   └── quick_retrain.py                 # Script-form training
│
├── HUS-Dissertation-Template-main (1)/  # LaTeX thesis template
├── docker-compose.yml
├── start.sh                             # Auto-detect Docker vs local
├── README.md
└── OVERVIEW.md                          # (file này)
```

---

## 4. Yêu cầu hệ thống

### Backend
- Python 3.10+
- RAM: tối thiểu 4 GB (8 GB khuyến nghị với GPU)
- GPU: tùy chọn — tự động detect CUDA nếu có, fallback về CPU

### Frontend
- Node.js 18+
- npm hoặc yarn

### Docker (tùy chọn)
- Docker Engine 24+
- Docker Compose v2+

---

## 5. Hướng dẫn cài đặt

### Cách 1: Khởi động nhanh (local, không Docker)

```bash
# Clone hoặc vào thư mục dự án
cd BrainSeg-XAI

# (Tùy chọn) Đặt model weights
mkdir -p backend/weights
cp your_model.pth backend/weights/unet_best.pth

# Chạy bằng script
chmod +x start.sh
./start.sh
```

Nếu không có file weights, hệ thống tự động chạy **demo mode** — pipeline đầy đủ nhưng kết quả phân đoạn không có ý nghĩa thực tế.

### Cách 2: Thủ công (backend + frontend riêng)

**Backend:**
```bash
cd backend
pip install -r requirements.txt
python3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
API docs tự động: http://localhost:8000/docs

**Frontend:**
```bash
cd frontend
npm install --legacy-peer-deps
npm start
```
Giao diện web: http://localhost:3000

### Cách 3: Docker Compose (production)

```bash
cp your_model.pth backend/weights/unet_best.pth
docker-compose up --build
```

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |

---

## 6. Cách sử dụng

### Qua giao diện web

1. Mở http://localhost:3000 trong trình duyệt.
2. Kéo thả hoặc click để chọn file ảnh MRI (PNG/JPG/TIFF/BMP/WEBP, tối đa 20 MB).
3. Đợi khoảng 1–3 giây (CPU) hoặc < 0.5 giây (GPU) để xử lý.
4. Xem kết quả trên 5 tab ảnh và bảng rủi ro bên phải.

### Qua API (cURL)

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -F "file=@/path/to/mri_brain.jpg" \
  | python3 -m json.tool
```

### Qua Python

```python
import requests

with open("mri_brain.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/predict",
        files={"file": ("mri.jpg", f, "image/jpeg")}
    )

data = response.json()
print(f"Risk: {data['risk']['risk_level']} ({data['risk']['risk_score']}/100)")
print(f"Tumor area: {data['features']['tumor_area_cm2']} cm²")
```

---

## 7. API Reference

### `GET /api/v1/health`

Kiểm tra trạng thái server.

**Response:**
```json
{ "status": "ok", "message": "Brain Tumor AI API đang hoạt động" }
```

### `POST /api/v1/predict`

Phân tích ảnh MRI và trả kết quả.

**Request:** `multipart/form-data` với trường `file` là file ảnh.

**Constraints:**
- Content-type: phải bắt đầu bằng `image/`
- Kích thước: ≤ 20 MB

**Status codes:**
| Code | Ý nghĩa |
|---|---|
| `200` | Thành công |
| `400` | File không phải ảnh |
| `413` | File quá lớn |
| `422` | Không đọc được ảnh |
| `500` | Lỗi model/inference |

---

## 8. Cấu trúc Response JSON

```json
{
  "original_b64":    "<base64 PNG>",
  "mask_b64":        "<base64 PNG — mặt nạ nhị phân màu xanh lá>",
  "overlay_b64":     "<base64 PNG — ảnh gốc + overlay đỏ vùng u>",
  "heatmap_b64":     "<base64 PNG — bản đồ nhiệt JET Grad-CAM>",
  "cam_overlay_b64": "<base64 PNG — ảnh gốc + heatmap blend>",

  "features": {
    "tumor_detected":      true,
    "tumor_area_px":       5420,
    "tumor_area_cm2":      47.64,
    "occupancy_ratio":     8.27,
    "num_regions":         1,
    "shape_irregularity":  0.42,
    "compactness":         0.58,
    "boundary_complexity": 11.3,
    "midline_shift":       false,
    "location":            "thùy đỉnh/thái dương, bên trái",
    "centroid_x":          98.5,
    "centroid_y":          110.2
  },

  "risk": {
    "risk_level":      "Trung bình",
    "risk_score":      38,
    "severity":        "Trung bình",
    "risk_color":      "#ffd60a",
    "fired_rules":     ["Tỷ lệ chiếm vùng não cao (8.27% > 5%)", "..."],
    "recommendations": ["Theo dõi sát và chụp MRI định kỳ 3 tháng/lần"],
    "explanation":     "Hệ thống phân tích 6 đặc trưng y khoa..."
  }
}
```

### Bảng đặc trưng hình thái

| Trường | Kiểu | Mô tả |
|---|---|---|
| `tumor_detected` | bool | Có phát hiện khối u không |
| `tumor_area_px` | int | Diện tích theo pixel |
| `tumor_area_cm2` | float | Diện tích ước tính (cm²) |
| `occupancy_ratio` | float | % diện tích não bị chiếm |
| `num_regions` | int | Số vùng liên thông riêng biệt |
| `shape_irregularity` | float | 0 = tròn đều, 1 = bất quy tắc |
| `compactness` | float | 1 - shape_irregularity |
| `boundary_complexity` | float | chu_vi / √diện_tích |
| `midline_shift` | bool | Tâm khối u lệch > 10% từ đường giữa |
| `location` | str | Vị trí giải phẫu ước tính |

### Phân loại mức rủi ro

| Điểm | Mức | Màu |
|---|---|---|
| 0 – 29 | Thấp | `#00c896` (xanh lá) |
| 30 – 49 | Trung bình | `#ffd60a` (vàng) |
| 50 – 69 | Cao | `#ff6b35` (cam) |
| 70 – 100 | Rất cao | `#ff2d55` (đỏ) |

---

## 9. Mô hình học sâu

### Kiến trúc hỗ trợ

Hệ thống tự động nhận dạng loại checkpoint khi tải:

| Loại mô hình | Nhận dạng bằng | Kiến trúc |
|---|---|---|
| Vanilla U-Net | Key bắt đầu bằng `inc.`, `down`, `up` | `models/unet.py::UNet` |
| SMP (U-Net++, v.v.) | Metadata `arch`/`encoder` trong checkpoint | `smp.create_model()` |

### Lưu checkpoint với metadata (khuyến nghị)

```python
torch.save({
    "arch":             "unetplusplus",
    "encoder":          "efficientnet-b4",
    "model_state_dict": model.state_dict(),
    "optimizer_state":  optimizer.state_dict(),
    "epoch":            epoch,
    "val_dice":         best_dice,
}, "backend/weights/unet_best.pth")
```

### Tiền xử lý ảnh

Tất cả ảnh đầu vào được chuẩn hóa về:
- Kích thước: `256 × 256` pixel (INTER_LINEAR)
- Chuẩn hóa: ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`
- Tensor: `[1, 3, 256, 256]` float32

---

## 10. Huấn luyện mô hình

### Dataset

**LGG MRI Segmentation** (Kaggle):
- URL: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation
- 3.929 ảnh MRI não 2D, 110 bệnh nhân LGG
- Ground truth mask nhị phân

Tải và giải nén vào `notebooks/data/lgg-mri-segmentation/`.

### Huấn luyện qua notebook

```bash
cd notebooks
jupyter notebook training.ipynb
```

Hoặc script:

```bash
python3 notebooks/quick_retrain.py
```

### Cấu hình huấn luyện điển hình

```python
ARCH    = "unetplusplus"
ENCODER = "efficientnet-b4"
LR      = 1e-4
EPOCHS  = 50
BATCH   = 16
LOSS    = "BCE + Dice"  # kết hợp để tốt hơn từng thành phần riêng lẻ
```

### Sau khi huấn luyện

```bash
cp path/to/best_model.pth backend/weights/unet_best.pth
```

Server sẽ tự động tải weights mới trong request tiếp theo (hoặc restart server để load ngay).

---

## 11. Giải thích kết quả (XAI)

### Grad-CAM

Grad-CAM sử dụng gradient của tín hiệu mục tiêu theo bản đồ đặc trưng tại layer cuối:

```
αₖ = (1/Z) Σᵢⱼ ∂(Σ σ(logit)) / ∂Aᵢⱼᵏ

L_GradCAM = ReLU(Σₖ αₖ · Aᵏ)
```

**Layer mục tiêu:**
- Vanilla UNet: `model.up4.conv.conv[-3]`
- SMP models: `model.segmentation_head[0]`

**Đọc bản đồ nhiệt:**
- Màu đỏ/vàng: vùng ảnh ảnh hưởng mạnh đến quyết định phân đoạn
- Màu xanh/tím: vùng ảnh ít ảnh hưởng
- Bản đồ nhiệt **không chỉ hiển thị khối u** mà hiển thị vùng ảnh hưởng đến quyết định tổng thể

### 5 chế độ xem ảnh

| Tab | Nội dung |
|---|---|
| Ảnh Gốc | MRI đầu vào sau resize 256×256 |
| Mask Overlay | Ảnh gốc + vùng khối u tô đỏ (alpha 45%) |
| Segmentation | Mặt nạ nhị phân — khối u màu xanh lá `#00ff88` |
| XAI Overlay | Ảnh gốc blend với Grad-CAM heatmap (alpha 50%) |
| Heatmap | Bản đồ nhiệt Grad-CAM thuần JET colormap |

---

## 12. Bộ máy luật y khoa

Sáu quy tắc độc lập cộng điểm vào tổng điểm rủi ro (cap 100):

| Quy tắc | Điều kiện | Điểm |
|---|---|---|
| R1: Diện tích chiếm | occupancy > 20% | +30 |
| | occupancy > 10% | +20 |
| | occupancy > 5% | +10 |
| R2: Bất đối xứng | irregularity > 0.7 | +25 |
| | irregularity > 0.5 | +15 |
| R3: Đường biên | boundary > 15 | +20 |
| | boundary > 10 | +10 |
| R4: Dịch đường giữa | midline_shift = True | +20 |
| R5: Nhiều vùng | num_regions > 3 | +15 |
| | num_regions > 1 | +8 |
| R6: Diện tích tuyệt đối | area > 10 cm² | +15 |
| | area > 4 cm² | +8 |

**Lưu ý:** Các ngưỡng và trọng số này dựa trên tham khảo tài liệu y văn và chưa được xác thực lâm sàng chính thức.

---

## 13. Câu hỏi thường gặp

**Q: Không có file weights thì sao?**
A: Hệ thống chạy "demo mode" với trọng số ngẫu nhiên. Pipeline đầy đủ (Grad-CAM, features, risk engine) đều hoạt động nhưng kết quả phân đoạn không có ý nghĩa thực tế.

**Q: Chạy trên GPU như thế nào?**
A: Hệ thống tự động detect CUDA: `DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")`. Không cần cấu hình thêm.

**Q: Hỗ trợ định dạng file nào?**
A: PNG, JPG/JPEG, BMP, TIFF, WEBP. Kích thước tùy ý (tự động resize về 256×256). Giới hạn 20 MB.

**Q: Làm sao thêm kiến trúc mô hình mới?**
A: Thêm kiến trúc vào SMP (hỗ trợ: `unet`, `unetplusplus`, `deeplabv3plus`, `fpn`, `manet`) bằng cách thay đổi `arch` khi lưu checkpoint. Để thêm kiến trúc hoàn toàn mới, chỉnh sửa `models/unet.py` và cập nhật logic nhận dạng trong `services/inference.py`.

**Q: Kết quả chậm trên CPU?**
A: Thời gian inference trên CPU (không GPU) khoảng 1–3 giây cho một ảnh. Grad-CAM yêu cầu một backward pass bổ sung, chiếm khoảng 40% tổng thời gian.

---

## 14. Tuyên bố từ chối trách nhiệm

Hệ thống BrainSeg-XAI được phát triển **chỉ cho mục đích nghiên cứu và học thuật**.

- Không được sử dụng như một công cụ chẩn đoán y tế trong thực hành lâm sàng.
- Mọi kết quả từ hệ thống cần được xác nhận bởi bác sĩ chuyên khoa thần kinh có chứng chỉ hành nghề.
- Bộ máy luật rủi ro chưa trải qua xác thực lâm sàng prospective.
- Tác giả không chịu trách nhiệm về bất kỳ quyết định y tế nào được đưa ra dựa trên đầu ra của hệ thống này.

---

*Tài liệu này được viết cho dự án BrainSeg-XAI — tiểu luận môn Học Máy trong Y tế, Khoa Vật lý, Trường Đại học Khoa học Tự nhiên, ĐHQGHN.*
