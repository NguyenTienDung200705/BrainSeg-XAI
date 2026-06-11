<div align="center">

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
<img src="https://img.shields.io/badge/React-18+-61DAFB?style=for-the-badge&logo=react&logoColor=black"/>
<img src="https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>

# 🧠 BrainAI

### Explainable Medical AI for Brain Tumor Segmentation

*Hệ thống AI phân tích và phân đoạn khối u não từ ảnh MRI với khả năng giải thích quyết định (XAI)*

[Demo](#-demo) · [Tính năng](#-tính-năng) · [Cài đặt](#-cài-đặt) · [API Docs](#-api-reference) · [Đóng góp](#-đóng-góp)

---

</div>

## 📋 Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Tính năng](#-tính-năng)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Yêu cầu hệ thống](#-yêu-cầu-hệ-thống)
- [Cài đặt](#-cài-đặt)
  - [Development](#1-development)
  - [Docker (Production)](#2-docker-production)
- [Dataset & Training](#-dataset--training)
- [API Reference](#-api-reference)
- [Công nghệ sử dụng](#-công-nghệ-sử-dụng)
- [Kết quả](#-kết-quả)
- [Đóng góp](#-đóng-góp)
- [Tác giả](#-tác-giả)
- [Giấy phép](#-giấy-phép)
- [Tuyên bố miễn trách](#️-tuyên-bố-miễn-trách)

---

## 🎯 Giới thiệu

**BrainAI** là hệ thống AI y tế hỗ trợ phân tích ảnh MRI não, tự động phát hiện và phân đoạn vùng khối u, đồng thời cung cấp khả năng **giải thích quyết định (Explainable AI)** thông qua Grad-CAM. Hệ thống được thiết kế hướng đến việc hỗ trợ bác sĩ chuyên khoa trong quá trình đọc phim và ra quyết định lâm sàng.

Dự án được xây dựng trên kiến trúc **U-Net** kết hợp với **Medical Rule Engine** để đánh giá mức độ nguy cơ, và triển khai dưới dạng web application với giao diện trực quan, dễ sử dụng.

> ⚠️ **Lưu ý quan trọng:** Hệ thống này chỉ dành cho mục đích nghiên cứu và học thuật. Không thay thế chẩn đoán y khoa chính thức.

---

## ✨ Tính năng

| Tính năng | Mô tả |
|-----------|-------|
| 🔍 **Phân đoạn khối u** | Tự động tạo binary mask xác định vùng khối u từ ảnh MRI |
| 📊 **Trích xuất đặc trưng** | Tính toán diện tích, tỉ lệ chiếm dụng, độ bất thường hình dạng, compactness |
| ⚕️ **Đánh giá nguy cơ** | Medical Rule Engine phân loại mức độ nguy cơ theo tiêu chí y khoa |
| 🌡️ **Grad-CAM XAI** | Heatmap trực quan hóa vùng ảnh ảnh hưởng đến quyết định của mô hình |
| 🖼️ **5-View Explorer** | Giao diện xem ảnh đa chiều: Original, Mask, Overlay, Heatmap, Side-by-side |
| 🐳 **Docker Ready** | Triển khai production đơn giản với Docker Compose |
| 📄 **API Docs** | Tự động sinh tài liệu API với Swagger UI / ReDoc |

---

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────┐
│                        BrainAI Pipeline                      │
└─────────────────────────────────────────────────────────────┘

  MRI Image Upload (JPEG / PNG / TIFF)
           │
           ▼
  ┌─────────────────┐
  │  Preprocessing  │  → Normalize (Z-score) · Resize 256×256 · RGB conversion
  └────────┬────────┘
           │
           ▼
  ┌──────────────────────┐
  │  U-Net Segmentation  │  → Encoder-Decoder · Skip Connections · Sigmoid output
  └──────────┬───────────┘
             │
      ┌──────┴──────┐
      │             │
      ▼             ▼
  ┌──────────┐  ┌──────────────────────┐
  │  Binary  │  │   Feature Extraction  │
  │   Mask   │  │  Area · Occupancy    │
  └──────────┘  │  Irregularity · etc. │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │  Medical Rule Engine │  → Risk Level: Low / Medium / High / Critical
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │     Grad-CAM XAI     │  → Heatmap · Saliency Map
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │   React Frontend     │  → 5-View Explorer · Risk Gauge · Report
                └──────────────────────┘
```

---

## 📂 Cấu trúc thư mục

```
brain-tumor-ai/
│
├── backend/                          # FastAPI application
│   ├── app/
│   │   ├── api/
│   │   │   └── predict.py            # API endpoints
│   │   ├── models/
│   │   │   └── unet.py               # U-Net architecture
│   │   ├── services/
│   │   │   ├── inference.py          # Model loading & prediction pipeline
│   │   │   └── feature_extraction.py # Morphological feature extraction
│   │   ├── rules/
│   │   │   └── medical_rules.py      # Medical Rule Engine (risk classification)
│   │   ├── xai/
│   │   │   └── gradcam.py            # Grad-CAM implementation
│   │   ├── utils/
│   │   │   └── image_utils.py        # Image preprocessing utilities
│   │   └── main.py                   # FastAPI app entry point
│   ├── weights/
│   │   └── unet_best.pth             # ← Đặt model weights tại đây
│   ├── tests/                        # Unit & integration tests
│   ├── Dockerfile
│   └── requirements.txt
│
├── frontend/                         # React application
│   ├── src/
│   │   ├── App.js                    # Root component & state management
│   │   ├── index.css                 # Design tokens & global styles
│   │   ├── components/
│   │   │   ├── NeuralCanvas.jsx      # Animated neural network background
│   │   │   ├── Header.jsx            # App header
│   │   │   ├── UploadZone.jsx        # Drag-and-drop MRI upload
│   │   │   ├── ProcessingPanel.jsx   # Real-time processing status
│   │   │   ├── ImageViewer.jsx       # 5-view image explorer
│   │   │   └── RiskPanel.jsx         # Risk gauge · Feature table · Rule output
│   │   └── services/
│   │       └── api.js                # Axios API client
│   ├── public/
│   ├── Dockerfile
│   └── nginx.conf
│
├── notebooks/
│   └── training.ipynb                # U-Net training notebook (Kaggle)
│
├── docker-compose.yml
├── .env.example
├── .gitignore
└── README.md
```

---

## 💻 Yêu cầu hệ thống

### Môi trường phát triển

| Thành phần | Phiên bản tối thiểu |
|------------|---------------------|
| Python | 3.10+ |
| Node.js | 18+ |
| CUDA *(tùy chọn)* | 11.8+ |
| RAM | 8 GB+ |
| GPU VRAM *(tùy chọn)* | 4 GB+ |

### Docker (Production)

| Thành phần | Phiên bản |
|------------|-----------|
| Docker | 24.0+ |
| Docker Compose | 2.20+ |

---

## 🚀 Cài đặt

### 1. Development

#### Clone repo

```bash
git clone https://github.com/your-username/brain-tumor-ai.git
cd brain-tumor-ai
```

#### Cấu hình môi trường

```bash
cp .env.example .env
# Chỉnh sửa .env theo cấu hình local của bạn
```

#### Backend (FastAPI)

```bash
cd backend

# Tạo virtual environment (khuyến nghị)
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# Cài đặt dependencies
pip install -r requirements.txt

# Đặt model weights
# Tải file .pth về và đặt vào:
mkdir -p weights
cp /path/to/your/model.pth weights/unet_best.pth

# Khởi động server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

> 📄 Swagger UI: http://localhost:8000/docs  
> 📄 ReDoc: http://localhost:8000/redoc

#### Frontend (React)

Mở terminal mới:

```bash
cd frontend
npm install --legacy-peer-deps
npm start
```

> 🌐 Web App: http://localhost:3000

---

### 2. Docker (Production)

```bash
# Đặt model weights trước
cp /path/to/your/model.pth backend/weights/unet_best.pth

# Build và khởi động toàn bộ stack
docker-compose up --build

# Chạy nền (detached mode)
docker-compose up --build -d

# Dừng
docker-compose down
```

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |

---

## 📊 Dataset & Training

### Dataset

**LGG MRI Segmentation** — Kaggle

| Thông tin | Chi tiết |
|-----------|----------|
| Link | [kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) |
| Số lượng ảnh | 3,929 ảnh MRI não |
| Loại mask | Binary segmentation masks |
| Nguồn gốc | The Cancer Genome Atlas (TCGA) |
| Bệnh lý | Low-Grade Glioma (LGG) |

### Training

Notebook training được cung cấp tại `notebooks/training.ipynb`, tương thích với Kaggle / Google Colab.

```bash
# Cài đặt dependencies training
pip install torch torchvision segmentation-models-pytorch albumentations

# Chạy notebook
jupyter notebook notebooks/training.ipynb
```

**Checkpoint formats được hỗ trợ:**

```python
# Format 1 — Raw state_dict
torch.save(model.state_dict(), "unet_best.pth")

# Format 2 — Dict với key model_state_dict
torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, "unet_best.pth")

# Format 3 — Dict với key state_dict
torch.save({"state_dict": model.state_dict()}, "unet_best.pth")
```

---

## 📡 API Reference

### `GET /api/v1/health`

Kiểm tra trạng thái API và model.

**Response:**

```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

### `POST /api/v1/predict`

Phân tích ảnh MRI và trả về kết quả phân đoạn, đặc trưng và đánh giá nguy cơ.

**Request:**

```
Content-Type: multipart/form-data
Body: file=<MRI image (JPEG/PNG/TIFF)>
```

**Response:**

```json
{
  "tumor_detected": true,
  "mask_base64": "<base64-encoded PNG>",
  "overlay_base64": "<base64-encoded PNG>",
  "heatmap_base64": "<base64-encoded PNG>",
  "features": {
    "tumor_area_px": 1842,
    "tumor_area_cm2": 4.2,
    "occupancy_ratio": 12.5,
    "shape_irregularity": 0.73,
    "compactness": 0.61,
    "mean_intensity": 187.4,
    "std_intensity": 32.1
  },
  "risk": {
    "risk_level": "High",
    "severity": "Severe",
    "triggered_rules": [
      "Tumor area exceeds 4 cm²",
      "Occupancy ratio > 10%"
    ],
    "recommendation": "Immediate specialist consultation recommended"
  }
}
```

**HTTP Status Codes:**

| Code | Mô tả |
|------|-------|
| `200` | Thành công |
| `400` | File không hợp lệ hoặc không phải ảnh MRI |
| `422` | Validation error |
| `500` | Lỗi inference (model hoặc server) |

---

## 🛠️ Công nghệ sử dụng

### AI / Deep Learning

| Thư viện | Mục đích |
|----------|----------|
| PyTorch | Framework deep learning chính |
| segmentation-models-pytorch | U-Net encoder-decoder |
| OpenCV | Xử lý ảnh và morphological operations |
| NumPy / SciPy | Tính toán numerical và feature extraction |

### Backend

| Thư viện | Mục đích |
|----------|----------|
| FastAPI | REST API framework |
| Uvicorn | ASGI server |
| Pillow | Đọc/ghi ảnh |
| python-multipart | Xử lý file upload |

### Frontend

| Thư viện | Mục đích |
|----------|----------|
| React 18 | UI framework |
| Axios | HTTP client |
| CSS Variables | Design system & theming |

### DevOps

| Công cụ | Mục đích |
|---------|----------|
| Docker | Container hóa ứng dụng |
| Docker Compose | Orchestration đa service |
| Nginx | Static file serving & reverse proxy |

---

## 📈 Kết quả

| Metric | Giá trị |
|--------|---------|
| Dice Score (validation) | ~0.87 |
| IoU | ~0.79 |
| Inference time (CPU) | ~1.2s / ảnh |
| Inference time (GPU) | ~0.15s / ảnh |

> *Kết quả có thể thay đổi tùy phiên bản model và cấu hình phần cứng.*

---
![Training Curve](HUS-Dissertation-Template-main%20(1)/HUS-Dissertation-Template-main/Figures/Chapter/Chapter6.png)

![Training Curve](HUS-Dissertation-Template-main%20(1)/HUS-Dissertation-Template-main/Figures/Chapter/Chapter7.png)

## 👨‍💻 Tác giả

**Nguyen Tien Dung**

Lĩnh vực chuyên môn: AI · Computer Vision · Medical Imaging · Explainable AI

---

## 📄 Giấy phép

Dự án này được phân phối dưới giấy phép [MIT License](LICENSE).

---

## ⚠️ Tuyên bố miễn trách

> Hệ thống BrainAI được phát triển **chỉ cho mục đích nghiên cứu và học thuật**.  
> **Không được sử dụng** để thay thế chẩn đoán y khoa chính thức.  
> Mọi kết quả phân tích cần được xác nhận và diễn giải bởi bác sĩ chuyên khoa có chuyên môn.  
> Tác giả không chịu trách nhiệm về bất kỳ quyết định y tế nào dựa trên output của hệ thống này.

---
