# 🤝 Contributing to BrainAI

Cảm ơn bạn đã quan tâm đến việc đóng góp cho **BrainAI**.

BrainAI là một dự án nghiên cứu mã nguồn mở (*open-source research project*) tập trung vào:
- Brain Tumor Segmentation
- Explainable AI (XAI)
- Medical Image Analysis
- AI Transparency in Healthcare

Chúng tôi luôn chào đón:
- Students
- Developers
- Researchers
- AI Enthusiasts

tham gia phát triển dự án.

---

# 🌟 Các lĩnh vực có thể đóng góp | Contribution Areas

## 🧠 AI & Deep Learning
- Improve segmentation models
- Tối ưu mô hình U-Net
- Improve Grad-CAM visualization
- Research new architectures

---

## 🩺 Medical Imaging
- MRI preprocessing
- Feature extraction
- Tumor analysis
- Medical rule systems

---

## 🌐 Frontend & UI/UX
- Improve dashboard visualization
- Responsive UI
- Better user interaction
- React optimization

---

## ⚙️ Backend & Infrastructure
- FastAPI optimization
- Docker deployment
- API improvements
- Performance optimization

---

## 📚 Documentation
- Improve README
- API documentation
- Tutorials
- Bug reports

---

# 🚀 Getting Started

## 1️⃣ Fork Repository

Fork repository này về GitHub account của bạn.

---

## 2️⃣ Clone Your Fork

```bash
git clone https://github.com/NguyenTienDung200705/BrainSeg-XAI.git
```

---

## 3️⃣ Create New Branch

```bash
git checkout -b feature/your-feature-name
```

Ví dụ:

```text
feature/add-gradcam
feature/improve-ui
fix/api-validation
docs/update-readme
```

---

# 🛠️ Development Setup

## Backend Setup

```bash
cd backend

pip install -r requirements.txt

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Frontend Setup

```bash
cd frontend

npm install --legacy-peer-deps

npm start
```

---

# 🧹 Coding Standards | Quy tắc coding

## Python
- Follow PEP8
- Đặt tên biến rõ ràng
- Keep functions modular
- Add comments for complex logic
- Use type hints when possible

---

## React
- Use reusable components
- Maintain responsive UI
- Keep clean architecture
- Avoid unnecessary complexity

---

# 📝 Git Commit Convention

Sử dụng commit message rõ ràng và có ý nghĩa.

## ✅ Good Examples

```text
feat: add Grad-CAM visualization
fix: improve API validation
docs: update README
refactor: optimize inference pipeline
```

---

# 📌 Pull Request Process

## Before Submitting PR

Vui lòng kiểm tra:

- [ ] Code chạy thành công
- [ ] Không chứa file không cần thiết
- [ ] Documentation đã được cập nhật
- [ ] Code được format hợp lý
- [ ] Không push dataset lớn
- [ ] Không push model weights
- [ ] Không chứa API keys hoặc secrets

---

## Create Pull Request

Khi tạo Pull Request, hãy mô tả:
- Nội dung thay đổi
- Lý do thay đổi
- Screenshots hoặc results nếu có

---

# 🐛 Reporting Issues

Nếu phát hiện lỗi:

1. Open GitHub Issue
2. Mô tả vấn đề rõ ràng
3. Include logs/screenshots nếu có
4. Provide reproduction steps

---

# 💡 Feature Requests

Mọi ý tưởng mới đều được hoan nghênh.

Possible future directions:
- 3D MRI segmentation
- Multi-class tumor segmentation
- Vision Transformers
- Better XAI methods
- Real-time inference optimization

---

# ❤️ Community Guidelines

Hãy giữ môi trường:
- Respectful
- Constructive
- Collaborative

trong:
- Discussions
- Pull Requests
- Code Reviews
- Issue Tracking

---

# ⚠️ Important Note

Dự án này phục vụ cho:
- Research
- Education
- Experimental AI Development

KHÔNG sử dụng thay thế chẩn đoán y khoa chuyên nghiệp (*clinical diagnosis*).

---

# 🙌 Thank You

Cảm ơn bạn đã đóng góp cho BrainAI.

Your contributions help advance:
- Medical AI
- Explainable AI
- Human-centered healthcare technology