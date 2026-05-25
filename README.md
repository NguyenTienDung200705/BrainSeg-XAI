# BrainAI — Explainable Medical AI for Brain Tumor Segmentation

Hệ thống AI phân tích khối u não từ ảnh MRI với khả năng giải thích quyết định (XAI).

---

##  Kiến trúc hệ thống

```
MRI Image Upload
      ↓
[Preprocessing]  → Normalize, Resize 256×256
      ↓
[U-Net Segmentation] → Binary mask (khối u / nền)
      ↓
[Feature Extraction] → Area, Occupancy, Shape Irregularity, Compactness...
      ↓
[Medical Rule Engine] → Đánh giá nguy cơ theo luật y khoa
      ↓
[Grad-CAM XAI] → Heatmap vùng ảnh quan trọng
      ↓
[Web Visualization] → Kết quả hiển thị trên React frontend
```

---

## Chạy nhanh (Development)

### 1. Backend (FastAPI)

```bash
cd backend
pip install -r requirements.txt

# backend/weights/unet_best.pth

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs: http://localhost:8000/docs

### 2. Frontend (React)

```bash
cd frontend
npm install --legacy-peer-deps
npm start
```

Web: http://localhost:3000

---

## Docker (Production)

<!-- ```bash
# Copy weights trước
cp your_model.pth backend/weights/unet_best.pth -->

# Build & run
docker-compose up --build

# Web: http://localhost:3000
# API: http://localhost:8000
```

<!-- ---

## Cấu trúc thư mục

```
brain-tumor-ai/
├── backend/
│   ├── app/
│   │   ├── api/predict.py          # FastAPI endpoints
│   │   ├── models/unet.py          # U-Net architecture
│   │   ├── services/
│   │   │   ├── inference.py        # Model loading & prediction
│   │   │   └── feature_extraction.py
│   │   ├── rules/medical_rules.py  # Medical Rule Engine
│   │   ├── xai/gradcam.py          # Grad-CAM implementation
│   │   ├── utils/image_utils.py    # Image processing
│   │   └── main.py                 # FastAPI app
│   ├── weights/unet_best.pth       # ← Đặt model của bạn vào đây
│   ├── Dockerfile
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── App.js                  # Main app logic
│   │   ├── index.css               # Global styles & design tokens
│   │   ├── components/
│   │   │   ├── NeuralCanvas.jsx    # Animated background
│   │   │   ├── Header.jsx
│   │   │   ├── UploadZone.jsx
│   │   │   ├── ProcessingPanel.jsx
│   │   │   ├── ImageViewer.jsx     # 5-view image explorer
│   │   │   └── RiskPanel.jsx       # Risk gauge + features + rules
│   │   └── services/api.js
│   ├── Dockerfile
│   └── nginx.conf
│
├── notebooks/
│   └── training.ipynb              # U-Net training notebook
│
└── docker-compose.yml -->
```

---

##  API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `GET`  | `/api/v1/health` | Kiểm tra trạng thái API |
| `POST` | `/api/v1/predict` | Upload MRI → nhận kết quả phân tích |

### Response từ `/api/v1/predict`

```json
{
  "original_b64": "...",
  "mask_b64": "...",
  "overlay_b64": "...",
  "heatmap_b64": "...",
  "cam_overlay_b64": "...",
  "features": {
    "tumor_detected": true,
    "tumor_area_cm2": 4.2,
    "occupancy_ratio": 12.5,
    "shape_irregularity": 0.65,
    "compactness": 0.35,
    "boundary_complexity": 11.2,
    "midline_shift": false,
    "location": "thùy đỉnh/thái dương, bên trái",
    "num_regions": 1
  },
  "risk": {
    "risk_level": "Cao",
    "risk_score": 55,
    "severity": "Nghiêm trọng",
    "risk_color": "#ff6b35",
    "fired_rules": ["..."],
    "recommendations": ["..."],
    "explanation": "..."
  }
}
```

---

Backend hỗ trợ 3 định dạng checkpoint:
- Raw `state_dict` (torch.save(model.state_dict(), ...))
- Dict với key `model_state_dict`
- Dict với key `state_dict`

---

## 📊 Dataset

**LGG MRI Segmentation** (Kaggle):
- Link: https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation
- 3,929 ảnh MRI não
- Ground truth segmentation masks

---

## ⚠️ Disclaimer

Hệ thống này chỉ dành cho **mục đích nghiên cứu và học thuật**.  
Không sử dụng thay thế chẩn đoán y khoa chính thức.  
Mọi kết quả cần được xác nhận bởi bác sĩ chuyên khoa.
