"""
Medical Q&A Chatbot API endpoint.
Uses Anthropic Claude when API key is available,
falls back to a rich local knowledge base otherwise.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import httpx
import os
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

SYSTEM_PROMPT = """Bạn là trợ lý AI y khoa chuyên biệt của hệ thống BrainAI - một nền tảng phân tích ảnh MRI não sử dụng AI.

NHIỆM VỤ CHÍNH:
1. Giải thích các khái niệm y khoa về khối u não, MRI, chẩn đoán hình ảnh
2. Giải thích cách hoạt động của các mô hình AI như U-Net, Grad-CAM, segmentation
3. Hỗ trợ người dùng hiểu kết quả phân tích từ hệ thống BrainAI
4. Trả lời câu hỏi về triệu chứng, phân loại khối u, quy trình điều trị
5. Cung cấp thông tin tham khảo về thần kinh học và ung thư học

NGUYÊN TẮC TRẢ LỜI:
- Luôn trả lời bằng tiếng Việt, rõ ràng và dễ hiểu
- Sử dụng ngôn ngữ chuyên môn nhưng giải thích đơn giản
- Cấu trúc câu trả lời có logic, dùng danh sách khi cần thiết
- Luôn nhấn mạnh: đây là thông tin tham khảo, không thay thế chẩn đoán bác sĩ
- Khi phát hiện câu hỏi về cấp cứu hoặc triệu chứng nghiêm trọng, khuyến cáo đến cơ sở y tế ngay

ĐỊNH DẠNG:
- Câu trả lời ngắn gọn (100-300 từ trừ khi cần giải thích phức tạp)
- Dùng emoji phù hợp (🧠 🔍 ⚕️ ⚠️)
- Kết thúc bằng lời khuyên hữu ích nếu phù hợp"""


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    history: Optional[List[ChatMessage]] = []
    analysis_context: Optional[str] = None   # ← dữ liệu kết quả phân tích từ frontend

class ChatResponse(BaseModel):
    answer: str
    model: str = "claude-sonnet-4-20250514"


# ---------------------------------------------------------------------------
# Rich local knowledge base  (used when no API key is present)
# Each entry: list of trigger keywords  →  answer text
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE = [
    # ── NGUY HIỂM / RỦI RO ──────────────────────────────────────────────
    {
        "keywords": ["nguy hiểm", "nguy cấp", "nguy hại", "có nguy", "nguy hiểm không",
                     "có sao không", "có lo không", "có nghiêm trọng", "nặng không",
                     "có cần phẫu", "cần mổ không", "cần điều trị", "phải làm gì"],
        "answer": (
            "⚠️ **Mức độ nguy hiểm của khối u não phụ thuộc vào nhiều yếu tố:**\n\n"
            "**Yếu tố quyết định:**\n"
            "• **Loại u**: U lành tính (meningioma, u tuyến yên) ít nguy hiểm hơn u ác tính (glioblastoma)\n"
            "• **Vị trí**: U gần vùng điều khiển hô hấp, tim mạch = nguy hiểm hơn\n"
            "• **Kích thước & tốc độ phát triển**: U lớn nhanh gây áp lực não nghiêm trọng\n"
            "• **Mức độ xâm lấn**: U xâm lấn mô lành xung quanh = khó điều trị hơn\n\n"
            "**Dấu hiệu cần đến cấp cứu ngay:**\n"
            "🚨 Đau đầu dữ dội đột ngột | Co giật lần đầu | Liệt nửa người đột ngột | Mất ý thức\n\n"
            "⚕️ Kết quả AI chỉ là bước sàng lọc ban đầu. Bác sĩ thần kinh sẽ đánh giá chính xác dựa trên toàn bộ bệnh sử và xét nghiệm bổ sung."
        )
    },
    # ── DIỆN TÍCH / KÍCH THƯỚC ──────────────────────────────────────────
    {
        "keywords": ["diện tích", "kích thước", "bao nhiêu cm", "bao nhiêu mm", "pixel",
                     "lớn không", "nhỏ không", "thể tích", "size", "kích cỡ"],
        "answer": (
            "📐 **Diện tích và kích thước khối u trong phân tích BrainAI:**\n\n"
            "Hệ thống tính diện tích dựa trên **số pixel vùng phân đoạn (segmentation mask)** từ mô hình U-Net.\n\n"
            "**Ý nghĩa kích thước:**\n"
            "• **< 2 cm**: U nhỏ, thường theo dõi định kỳ hoặc phẫu thuật nội soi\n"
            "• **2–4 cm**: U trung bình, cần đánh giá kỹ vị trí và loại u\n"
            "• **> 4 cm**: U lớn, thường có triệu chứng rõ, cần can thiệp\n\n"
            "**Lưu ý quan trọng:**\n"
            "Kích thước trên ảnh MRI 2D chỉ là một lát cắt. Thể tích thực 3D cần nhiều lát cắt liên tiếp.\n"
            "Kết quả diện tích pixel của hệ thống cần bác sĩ chuyển đổi sang đơn vị thực (cm²) dựa trên thông số máy MRI.\n\n"
            "⚕️ Hãy mang kết quả đến bác sĩ để được tư vấn đầy đủ."
        )
    },
    # ── U LÀNH / U ÁC ───────────────────────────────────────────────────
    {
        "keywords": ["lành tính", "ác tính", "u lành", "u ác", "benign", "malignant",
                     "ung thư não", "cancer", "grade", "độ ác"],
        "answer": (
            "🧠 **Phân biệt u lành tính và u ác tính não:**\n\n"
            "| Đặc điểm | U lành tính | U ác tính |\n"
            "|---|---|---|\n"
            "| Tốc độ phát triển | Chậm | Nhanh |\n"
            "| Ranh giới | Rõ ràng | Mờ, xâm lấn |\n"
            "| Di căn | Không | Có thể |\n"
            "| Tiên lượng | Tốt hơn | Xấu hơn |\n\n"
            "**Phân loại WHO (Grade I–IV):**\n"
            "• Grade I–II: Tăng trưởng chậm, tiên lượng tốt\n"
            "• Grade III–IV: Ác tính cao, cần điều trị tích cực\n\n"
            "**Ví dụ:**\n"
            "• Lành: Meningioma, Acoustic neuroma, U tuyến yên\n"
            "• Ác: Glioblastoma (GBM), Anaplastic astrocytoma\n\n"
            "⚕️ Chẩn đoán chính xác cần sinh thiết mô học, không thể xác định chỉ qua MRI."
        )
    },
    # ── TRIỆU CHỨNG ─────────────────────────────────────────────────────
    {
        "keywords": ["triệu chứng", "dấu hiệu", "biểu hiện", "đau đầu", "buồn nôn",
                     "co giật", "mờ mắt", "thị lực", "nôn mửa", "chóng mặt",
                     "yếu tay", "yếu chân", "tê liệt", "mất trí", "quên"],
        "answer": (
            "🔍 **Triệu chứng thường gặp của khối u não:**\n\n"
            "**Triệu chứng phổ biến:**\n"
            "• 🤕 Đau đầu dai dẳng, thường nặng hơn vào buổi sáng\n"
            "• 🌀 Buồn nôn, nôn mửa không rõ nguyên nhân\n"
            "• 👁️ Mờ mắt, nhìn đôi hoặc mất thị lực một bên\n"
            "• ⚡ Co giật lần đầu ở người trưởng thành\n"
            "• 🧠 Thay đổi tính cách, trí nhớ giảm sút\n"
            "• 💪 Yếu, tê liệt tay/chân một bên\n\n"
            "**Triệu chứng theo vị trí u:**\n"
            "• Thùy trán: Thay đổi tính cách, khó nói\n"
            "• Thùy thái dương: Rối loạn ngôn ngữ, trí nhớ\n"
            "• Thùy chẩm: Rối loạn thị giác\n"
            "• Tiểu não: Mất thăng bằng, đi loạng choạng\n\n"
            "🚨 **Nếu có co giật đột ngột, liệt người, hoặc đau đầu dữ dội: đến cấp cứu ngay!**"
        )
    },
    # ── ĐIỀU TRỊ ────────────────────────────────────────────────────────
    {
        "keywords": ["điều trị", "chữa trị", "phẫu thuật", "mổ", "xạ trị", "hóa trị",
                     "thuốc", "tiên lượng", "sống được bao lâu", "khỏi không",
                     "tỷ lệ sống", "treatment", "therapy"],
        "answer": (
            "⚕️ **Các phương pháp điều trị khối u não:**\n\n"
            "**1. Phẫu thuật (Surgical resection)**\n"
            "• Mục tiêu: Lấy toàn bộ hoặc một phần u\n"
            "• Áp dụng khi u có thể tiếp cận và không ở vùng nguy hiểm\n\n"
            "**2. Xạ trị (Radiation therapy)**\n"
            "• Gamma Knife: Xạ phẫu định vị, chính xác cao\n"
            "• Xạ trị toàn não: Cho u di căn nhiều vị trí\n\n"
            "**3. Hóa trị (Chemotherapy)**\n"
            "• Temozolomide: Thuốc phổ biến cho glioblastoma\n"
            "• Thường kết hợp với xạ trị\n\n"
            "**4. Liệu pháp mới:**\n"
            "• Liệu pháp miễn dịch (Immunotherapy)\n"
            "• Điều trị đích (Targeted therapy)\n"
            "• Tumor Treating Fields (TTFields)\n\n"
            "**Tiên lượng** phụ thuộc vào: loại u, grade, tuổi bệnh nhân, vị trí và đáp ứng điều trị.\n\n"
            "⚕️ Bác sĩ ung bướu thần kinh sẽ đưa ra phác đồ phù hợp nhất."
        )
    },
    # ── UNET ────────────────────────────────────────────────────────────
    {
        "keywords": ["u-net", "unet", "mạng nơ ron", "deep learning", "segmentation",
                     "phân đoạn", "mô hình ai", "neural network", "cnn", "model"],
        "answer": (
            "🤖 **U-Net - Mô hình AI phân đoạn ảnh y tế:**\n\n"
            "U-Net là kiến trúc mạng nơ-ron tích chập (CNN) được thiết kế đặc biệt cho **phân đoạn ảnh y tế**.\n\n"
            "**Cấu trúc gồm 2 phần:**\n"
            "• **Encoder (nhánh trái)**: Thu nhỏ ảnh, trích xuất đặc trưng (features)\n"
            "• **Decoder (nhánh phải)**: Phóng to lại, tái tạo mask phân đoạn\n"
            "• **Skip connections**: Kết nối encoder-decoder, giữ thông tin vị trí\n\n"
            "**Tại sao U-Net tốt cho y tế?**\n"
            "✅ Hoạt động tốt với ít dữ liệu huấn luyện\n"
            "✅ Phân đoạn pixel chính xác, biên rõ ràng\n"
            "✅ Nhẹ, chạy nhanh\n\n"
            "**Trong BrainAI:** U-Net với encoder EfficientNet-B4 phân tích ảnh MRI để xác định vùng khối u."
        )
    },
    # ── GRAD-CAM / XAI ──────────────────────────────────────────────────
    {
        "keywords": ["grad-cam", "gradcam", "xai", "explainable", "giải thích",
                     "heatmap", "bản đồ nhiệt", "minh bạch", "vùng nào", "ai nhìn vào đâu"],
        "answer": (
            "🔍 **Grad-CAM - AI giải thích quyết định của mình:**\n\n"
            "Grad-CAM (Gradient-weighted Class Activation Mapping) là kỹ thuật **XAI (Explainable AI)** - giúp AI 'nói' nó đang nhìn vào đâu.\n\n"
            "**Cách hoạt động:**\n"
            "1. AI phân tích ảnh MRI → đưa ra dự đoán\n"
            "2. Grad-CAM tính gradient ngược từ lớp cuối\n"
            "3. Tạo **bản đồ nhiệt (heatmap)** overlay lên ảnh gốc\n"
            "4. Màu đỏ/vàng = AI quan tâm nhiều nhất\n\n"
            "**Tại sao quan trọng trong y tế?**\n"
            "• Bác sĩ có thể kiểm chứng AI có nhìn đúng vùng khối u không\n"
            "• Phát hiện AI đang 'lừa dối' (nhìn vào artifact thay vì u thật)\n"
            "• Tăng tin tưởng của bác sĩ vào hệ thống AI\n\n"
            "💡 BrainAI hiển thị Grad-CAM overlay để bác sĩ và người dùng hiểu cơ sở quyết định của AI."
        )
    },
    # ── KẾT QUẢ PHÂN TÍCH ───────────────────────────────────────────────
    {
        "keywords": ["kết quả", "phân tích", "báo cáo", "risk", "rủi ro", "score",
                     "điểm số", "accuracy", "độ chính xác", "confidence", "xác suất"],
        "answer": (
            "📋 **Hiểu kết quả phân tích của BrainAI:**\n\n"
            "**Các chỉ số trong báo cáo:**\n\n"
            "🔴 **Risk Score (Điểm rủi ro)**: 0–100\n"
            "• 0–30: Rủi ro thấp\n"
            "• 30–60: Rủi ro trung bình, cần theo dõi\n"
            "• 60–100: Rủi ro cao, cần khám chuyên khoa\n\n"
            "📐 **Segmentation Mask**: Vùng AI xác định là bất thường\n\n"
            "🌡️ **Grad-CAM Heatmap**: Vùng AI 'nhìn vào' nhiều nhất\n\n"
            "📊 **Features**: Các đặc trưng trích xuất (diện tích, cường độ, hình dạng)\n\n"
            "**Lưu ý quan trọng:**\n"
            "⚠️ AI có thể cho kết quả dương tính giả (false positive) hoặc âm tính giả (false negative). Kết quả cần được bác sĩ xác nhận bằng đọc ảnh MRI trực tiếp và khám lâm sàng."
        )
    },
    # ── MENINGIOMA ──────────────────────────────────────────────────────
    {
        "keywords": ["meningioma", "màng não", "u màng não"],
        "answer": (
            "🧠 **Meningioma - U màng não:**\n\n"
            "Meningioma là u phát sinh từ **màng não (meninges)** - lớp bọc bên ngoài não và tủy sống.\n\n"
            "**Đặc điểm:**\n"
            "• Chiếm ~30% u não nguyên phát\n"
            "• Phần lớn lành tính (Grade I), phát triển chậm\n"
            "• Phổ biến ở phụ nữ trung niên (40–60 tuổi)\n\n"
            "**Triệu chứng** (phụ thuộc vị trí):\n"
            "• Đau đầu tăng dần\n"
            "• Yếu/tê tay chân\n"
            "• Rối loạn thị giác\n"
            "• Có thể không có triệu chứng nhiều năm\n\n"
            "**Điều trị:**\n"
            "• Nhỏ, không triệu chứng: Theo dõi định kỳ\n"
            "• Có triệu chứng: Phẫu thuật ± xạ trị\n\n"
            "✅ Tiên lượng thường tốt nếu phát hiện sớm và điều trị kịp thời."
        )
    },
    # ── GLIOMA / GLIOBLASTOMA ────────────────────────────────────────────
    {
        "keywords": ["glioma", "glioblastoma", "gbm", "astrocytoma", "oligodendroglioma"],
        "answer": (
            "🧠 **Glioma - U thần kinh đệm:**\n\n"
            "Glioma xuất phát từ **tế bào thần kinh đệm (glial cells)** trong não và tủy sống.\n\n"
            "**Phân loại chính:**\n"
            "• **Astrocytoma**: Từ tế bào hình sao, Grade I–IV\n"
            "• **Oligodendroglioma**: Tiên lượng tốt hơn, đáp ứng hóa trị tốt\n"
            "• **Glioblastoma (GBM - Grade IV)**: Hung hãn nhất, sống trung bình 14–16 tháng\n\n"
            "**Đặc điểm GBM:**\n"
            "• Phát triển rất nhanh\n"
            "• Ranh giới không rõ, xâm lấn mô lành\n"
            "• Kháng trị cao\n\n"
            "**Điều trị chuẩn:**\n"
            "Phẫu thuật + Xạ trị + Temozolomide (hóa trị)\n\n"
            "⚕️ Nghiên cứu về liệu pháp miễn dịch và điều trị gen đang được tiến hành tích cực."
        )
    },
    # ── PHÂN TÍCH MRI ────────────────────────────────────────────────────
    {
        "keywords": ["mri", "cộng hưởng từ", "chụp mri", "ảnh mri", "t1", "t2",
                     "flair", "contrast", "cản quang", "đọc mri", "kết quả mri"],
        "answer": (
            "🔍 **Phân tích ảnh MRI não:**\n\n"
            "**Các chuỗi xung MRI thường dùng:**\n\n"
            "📷 **T1 (Không contrast)**\n"
            "• Mô mỡ: Sáng | Nước/CSF: Tối\n"
            "• Thấy rõ giải phẫu\n\n"
            "📷 **T1 + Contrast (Gadolinium)**\n"
            "• Vùng có hàng rào máu-não bị phá vỡ sẽ sáng lên\n"
            "• Xác định vùng u hoạt động, viêm\n\n"
            "📷 **T2 / FLAIR**\n"
            "• Nước/phù nề: Sáng\n"
            "• Phát hiện phù não quanh khối u\n\n"
            "📷 **DWI (Diffusion)**\n"
            "• Phát hiện nhồi máu não cấp\n\n"
            "**BrainAI** phân tích ảnh T1/T2, áp dụng U-Net để phân đoạn vùng bất thường và Grad-CAM để trực quan hóa."
        )
    },
    # ── THEO DÕI / TÁI KHÁM ─────────────────────────────────────────────
    {
        "keywords": ["theo dõi", "tái khám", "bao lâu khám lại", "định kỳ", "chụp lại",
                     "kiểm tra lại", "follow up", "mri lại"],
        "answer": (
            "📅 **Lịch theo dõi khối u não:**\n\n"
            "**Tần suất tái khám phụ thuộc vào loại u:**\n\n"
            "🟢 **U lành tính nhỏ, không triệu chứng** (ví dụ meningioma nhỏ):\n"
            "• MRI mỗi 6–12 tháng năm đầu\n"
            "• Sau đó mỗi 1–2 năm nếu ổn định\n\n"
            "🟡 **Sau phẫu thuật u lành tính:**\n"
            "• MRI sau mổ 3–6 tháng\n"
            "• Theo dõi hàng năm\n\n"
            "🔴 **Sau điều trị u ác tính (glioma, GBM):**\n"
            "• MRI mỗi 2–3 tháng trong 2 năm đầu\n"
            "• Cần phân biệt tái phát vs. radiation necrosis\n\n"
            "**Dấu hiệu cần tái khám ngay (không chờ lịch):**\n"
            "🚨 Triệu chứng mới xuất hiện | Co giật | Đau đầu tăng đột ngột\n\n"
            "⚕️ Bác sĩ sẽ điều chỉnh lịch theo dõi phù hợp với từng bệnh nhân."
        )
    },
    # ── CÁC ĐẶC TRƯNG TRONG PHÂN TÍCH ──────────────────────────────────
    {
        "keywords": ["đặc trưng", "feature", "feature extraction", "trích xuất",
                     "intensity", "cường độ", "texture", "hình dạng", "shape"],
        "answer": (
            "📊 **Feature Extraction trong BrainAI:**\n\n"
            "Sau khi U-Net tạo segmentation mask, hệ thống trích xuất các đặc trưng định lượng:\n\n"
            "**Đặc trưng hình dạng (Shape):**\n"
            "• Diện tích vùng bất thường (pixels)\n"
            "• Chu vi, độ tròn (circularity)\n"
            "• Hộp bao (bounding box)\n\n"
            "**Đặc trưng cường độ (Intensity):**\n"
            "• Giá trị pixel trung bình, max, min\n"
            "• Độ lệch chuẩn (texture thô)\n"
            "• Histogram phân bố cường độ\n\n"
            "**Đặc trưng kết cấu (Texture):**\n"
            "• Đồng nhất, độ tương phản\n"
            "• Entropy (độ phức tạp vùng u)\n\n"
            "**Medical Rule Engine** dùng các đặc trưng này để tính Risk Score và đưa ra khuyến nghị lâm sàng."
        )
    },
    # ── HỆ THỐNG BRAINAI ─────────────────────────────────────────────────
    {
        "keywords": ["brainai", "hệ thống", "hoạt động như thế nào", "pipeline",
                     "quy trình", "bước", "cách dùng", "sử dụng"],
        "answer": (
            "🤖 **Hệ thống BrainAI hoạt động như thế nào?**\n\n"
            "**Pipeline 6 bước:**\n\n"
            "**1️⃣ Tiền xử lý ảnh**\n"
            "Chuẩn hóa kích thước, contrast, loại nhiễu\n\n"
            "**2️⃣ U-Net Segmentation**\n"
            "Mô hình AI xác định và phân đoạn vùng bất thường\n\n"
            "**3️⃣ Feature Extraction**\n"
            "Trích xuất đặc trưng định lượng: diện tích, cường độ, texture\n\n"
            "**4️⃣ Medical Rule Engine**\n"
            "Hệ thống quy tắc lâm sàng tính Risk Score\n\n"
            "**5️⃣ Grad-CAM Visualization**\n"
            "Tạo heatmap giải thích vùng AI tập trung\n\n"
            "**6️⃣ Báo cáo kết quả**\n"
            "Tổng hợp phân tích, khuyến nghị lâm sàng\n\n"
            "⚠️ BrainAI là công cụ hỗ trợ chẩn đoán, không thay thế bác sĩ chuyên khoa."
        )
    },
]


# ---------------------------------------------------------------------------
# Parse context string → dict
# ---------------------------------------------------------------------------
def parse_context(ctx):
    if not ctx:
        return {}
    data = {}
    for line in ctx.splitlines():
        if ":" in line and not line.startswith("===") and not line.startswith("---"):
            key, _, val = line.partition(":")
            data[key.strip()] = val.strip()
    return data


# ---------------------------------------------------------------------------
# Fallback thông minh – đọc ĐÚNG tên field từ inference.py
# ---------------------------------------------------------------------------
def get_fallback_answer(question, analysis_context=None):
    q = question.lower().replace("?","").replace("!","").replace(",","")
    ctx = parse_context(analysis_context)
    has = bool(ctx)

    if has:
        risk_score   = ctx.get("Risk Score", "")
        risk_level   = ctx.get("Mức độ", "")
        severity     = ctx.get("Mức độ nghiêm trọng", "")
        explanation  = ctx.get("Giải thích", "")
        fired_rules  = ctx.get("Quy tắc kích hoạt", "")
        recommend    = ctx.get("Khuyến nghị", "")
        detected     = ctx.get("Phát hiện u", "")
        area_px      = ctx.get("Diện tích u", "").replace("pixels","").strip()
        area_cm2     = ctx.get("Diện tích u (cm²)", "").replace("cm²","").strip()
        occupancy    = ctx.get("Tỷ lệ chiếm não", "")
        num_regions  = ctx.get("Số vùng u", "")
        irregularity = ctx.get("Độ bất đối xứng", "")
        compactness  = ctx.get("Độ tròn (compactness)", "")
        boundary     = ctx.get("Phức tạp đường biên", "")
        midline      = ctx.get("Midline Shift", "")
        location     = ctx.get("Vị trí u", "")

        try:
            score_num = float(risk_score.split("/")[0])
        except Exception:
            score_num = None

        def risk_advice(s):
            if s is None: return "Vui lòng gặp bác sĩ để đánh giá."
            if s >= 70: return "KHẨN CẤP: Cần nhập viện và can thiệp ngay lập tức."
            if s >= 50: return "Cần điều trị chuyên khoa trong vòng 48 giờ."
            if s >= 30: return "Theo dõi và hội chẩn bác sĩ thần kinh trong 1–4 tuần."
            return "Theo dõi định kỳ, chụp MRI kiểm tra sau 6–12 tháng."

        def risk_emoji(s):
            if s is None: return "⚠️"
            if s >= 70: return "🔴 Rất cao – Nguy kịch"
            if s >= 50: return "🟠 Cao – Nghiêm trọng"
            if s >= 30: return "🟡 Trung bình"
            return "🟢 Thấp"

        def build(title, items, extra=""):
            parts = [title + "\n"]
            for item in items:
                if item:
                    parts.append(item)
            if extra:
                parts.append(extra)
            return "\n".join(parts)

        # ── giải thích số liệu cụ thể (ưu tiên cao nhất) ──────────────────
        # Người dùng hỏi về một con số họ thấy trên màn hình
        import re
        numbers_in_q = re.findall(r"\d+\.?\d*", q)
        explain_kws = ["tại sao","vì sao","nghĩa là gì","có nghĩa","giải thích",
                       "tính ra","sao lại","được tính","ý nghĩa","explain"]
        is_explaining_number = bool(numbers_in_q) and any(k in q for k in explain_kws)

        # Match số trong câu hỏi với số liệu thực tế
        def num_matches(val_str, user_nums):
            if not val_str: return False
            try:
                v = str(float(str(val_str).split()[0]))
                return any(abs(float(n) - float(v)) < 0.01 for n in user_nums)
            except Exception:
                return False

        if is_explaining_number or any(k in q for k in ["tính thế nào","công thức","được đo"]):
            # Xác định số nào user đang hỏi
            matched_field = None
            matched_val = None
            field_map = [
                (irregularity, "shape_irregularity", "Độ bất đối xứng (shape_irregularity)"),
                (compactness,  "compactness",        "Độ tròn (compactness)"),
                (boundary,     "boundary_complexity","Phức tạp đường biên (boundary_complexity)"),
                (area_cm2,     "tumor_area_cm2",     "Diện tích u (cm²)"),
                (occupancy,    "occupancy_ratio",    "Tỷ lệ chiếm não (%)"),
                (risk_score.split("/")[0] if risk_score else "", "risk_score", "Risk Score"),
            ]
            for val, field, label in field_map:
                if num_matches(val, numbers_in_q):
                    matched_field = label
                    matched_val = val
                    break

            if matched_field == "Độ bất đối xứng (shape_irregularity)":
                v = float(matched_val) if matched_val else 0
                level = "rất cao (nguy cơ ác tính tăng)" if v > 0.7 else "trung bình" if v > 0.5 else "thấp (hình dạng tương đối đều)"
                msg = (
                    "🔬 **Giải thích độ bất đối xứng = {}:**\n\n"
                    "**Cách tính:**\n"
                    "shape_irregularity = 1 - compactness\n"
                    "compactness = (4π × diện tích) / (chu vi²)\n\n"
                    "**Thang đo:**\n"
                    "• = 0 → hình tròn hoàn hảo (lý tưởng)\n"
                    "• = 1 → hoàn toàn bất đối xứng\n"
                    "• {} → mức {}\n\n"
                    "**Ý nghĩa lâm sàng:**\n"
                    "• < 0.5 → Hình dạng đều, thường gặp ở u lành tính\n"
                    "• 0.5–0.7 → Bất đối xứng vừa, cần theo dõi\n"
                    "• > 0.7 → Bất đối xứng cao, gợi ý u ác tính\n\n"
                    "⚕️ Chỉ số này đóng góp 0–25 điểm vào Risk Score của hệ thống BrainAI."
                ).format(matched_val, matched_val, level)
                return msg

            elif matched_field == "Độ tròn (compactness)":
                v = float(matched_val) if matched_val else 0
                level = "rất tốt (u tròn đều)" if v > 0.8 else "trung bình" if v > 0.6 else "thấp (u không đều)"
                msg = (
                    "🔬 **Giải thích độ tròn (compactness) = {}:**\n\n"
                    "**Cách tính:**\n"
                    "compactness = (4π × diện tích) / (chu vi²)\n\n"
                    "**Thang đo:**\n"
                    "• = 1.0 → hình tròn hoàn hảo\n"
                    "• < 1.0 → càng thấp càng bất thường\n"
                    "• {} → {}\n\n"
                    "**Ý nghĩa lâm sàng:**\n"
                    "• > 0.8 → Hình dạng tốt, thường u lành tính\n"
                    "• 0.5–0.8 → Cần đánh giá thêm\n"
                    "• < 0.5 → Hình dạng bất thường, gợi ý ác tính\n\n"
                    "⚕️ Kết hợp với irregularity={} để đánh giá toàn diện."
                ).format(matched_val, matched_val, level, irregularity or "N/A")
                return msg

            elif matched_field == "Phức tạp đường biên (boundary_complexity)":
                v = float(matched_val) if matched_val else 0
                level = "rất phức tạp (nghi xâm lấn)" if v > 15 else "phức tạp vừa" if v > 10 else "tương đối đơn giản"
                msg = (
                    "🔬 **Giải thích phức tạp đường biên = {}:**\n\n"
                    "**Cách tính:**\n"
                    "boundary_complexity = chu vi / √(diện tích)\n\n"
                    "**Ý nghĩa:**\n"
                    "• Số cao → đường biên gồ ghề, gai góc\n"
                    "• Số thấp → đường biên trơn, mịn\n"
                    "• {} → {}\n\n"
                    "**Phân loại:**\n"
                    "• < 10 → Đường biên đơn giản\n"
                    "• 10–15 → Phức tạp vừa, theo dõi\n"
                    "• > 15 → Rất phức tạp, gợi ý xâm lấn mô lành\n\n"
                    "⚕️ Đây là dấu hiệu quan trọng phân biệt u lành/ác tính."
                ).format(matched_val, matched_val, level)
                return msg

            elif matched_field == "Tỷ lệ chiếm não (%)":
                v = float(matched_val) if matched_val else 0
                level = "rất cao" if v > 20 else "cao" if v > 10 else "trung bình" if v > 5 else "thấp"
                msg = (
                    "📐 **Giải thích tỷ lệ chiếm não = {}%:**\n\n"
                    "**Cách tính:**\n"
                    "occupancy_ratio = (pixels khối u / tổng pixels ảnh) × 100\n\n"
                    "**Ý nghĩa:**\n"
                    "{}% diện tích lát cắt MRI bị khối u chiếm → mức {}\n\n"
                    "**Phân loại lâm sàng:**\n"
                    "• < 5% → Thấp: theo dõi định kỳ\n"
                    "• 5–10% → Trung bình: cần đánh giá thêm\n"
                    "• 10–20% → Cao: hội chẩn chuyên khoa ngay\n"
                    "• > 20% → Rất cao: can thiệp khẩn cấp\n\n"
                    "⚕️ Chỉ số này đóng góp 0–30 điểm vào Risk Score."
                ).format(matched_val, matched_val, level)
                return msg

            elif matched_field == "Risk Score":
                recs_short = fired_rules[:120] + "..." if len(fired_rules) > 120 else fired_rules
                msg = (
                    "📊 **Giải thích Risk Score = {}:**\n\n"
                    "**Cách tính – tổng điểm từ 6 luật y khoa:**\n"
                    "• Tỷ lệ chiếm não: 0–30 điểm\n"
                    "• Độ bất đối xứng: 0–25 điểm\n"
                    "• Phức tạp đường biên: 0–20 điểm\n"
                    "• Midline Shift: 0–20 điểm\n"
                    "• Số vùng u: 0–15 điểm\n"
                    "• Kích thước tuyệt đối: 0–15 điểm\n\n"
                    "**Phân loại:**\n"
                    "• 0–29 → Thấp 🟢   • 30–49 → Trung bình 🟡\n"
                    "• 50–69 → Cao 🟠   • 70–100 → Rất cao 🔴\n\n"
                    "**Kết quả của bạn: {}/100 → {} ({})**\n\n"
                    "🔍 Luật kích hoạt: {}"
                ).format(matched_val, risk_score,
                         risk_level or "N/A", severity or "N/A", recs_short or "Xem kết quả phân tích")
                return msg

            # Có số nhưng không match field nào → tóm tắt
            if is_explaining_number:
                parts = ["📋 **Các chỉ số trong kết quả phân tích của bạn:**\n"]
                if risk_score:    parts.append("• **Risk Score:** {}".format(risk_score))
                if irregularity:  parts.append("• **Độ bất đối xứng:** {} (0=tròn đều, 1=bất đối xứng hoàn toàn)".format(irregularity))
                if compactness:   parts.append("• **Độ tròn (compactness):** {} (1=tròn hoàn hảo)".format(compactness))
                if boundary:      parts.append("• **Biên phức tạp:** {} (cao=gai góc)".format(boundary))
                if area_cm2:      parts.append("• **Diện tích u:** {} cm²".format(area_cm2))
                if occupancy:     parts.append("• **Tỷ lệ chiếm não:** {}%".format(occupancy))
                parts.append("\nHỏi cụ thể hơn, ví dụ: \'Tại sao độ bất đối xứng là 0.54?\'")
                return "\n".join(parts)

        # ── nguy hiểm / risk ────────────────────────────────────────────
        danger_kws = ["nguy hiểm","có sao","lo không","nghiêm trọng","mức độ",
                      "nặng không","đáng lo","cần mổ","cần điều trị","có nguy",
                      "risk","phải làm","nên làm","có cần","đánh giá"]
        if any(k in q for k in danger_kws):
            items = [
                "• **Risk Score:** {} → {}".format(risk_score, risk_emoji(score_num)) if risk_score else "",
                "• **Phân cấp nguy cơ:** {}".format(risk_level) if risk_level else "",
                "• **Mức độ nghiêm trọng:** {}".format(severity) if severity else "",
                "• **Phát hiện khối u:** {}".format(detected) if detected else "",
                "• **Midline Shift:** {}".format(midline) if midline else "",
                "• **Số vùng u:** {}".format(num_regions) if num_regions else "",
            ]
            rules_txt = ""
            if fired_rules:
                rule_list = [r.strip() for r in fired_rules.split("|") if r.strip()]
                rules_txt = "\n🔍 **Dấu hiệu nguy cơ:\n" + "\n".join("  • " + r for r in rule_list)
            recs_txt = ""
            if recommend:
                rec_list = [r.strip() for r in recommend.split("|") if r.strip()]
                recs_txt = "\n💡 **Khuyến nghị:**\n" + "\n".join("  • " + r for r in rec_list)
            footer = "\n⚕️ **Lời khuyên:** {}\n\n⚠️ Kết quả AI – bác sĩ xác nhận chẩn đoán cuối cùng.".format(risk_advice(score_num))
            return build("📋 **Đánh giá nguy cơ từ kết quả BrainAI:**",
                         [i for i in items if i], rules_txt + recs_txt + footer)

        # ── diện tích / kích thước ──────────────────────────────────────
        size_kws = ["diện tích","kích thước","bao nhiêu","pixel","cm",
                    "lớn không","nhỏ không","chiếm","size"]
        if any(k in q for k in size_kws):
            items = [
                "• **Diện tích u:** {} pixels".format(area_px) if area_px else "",
                "• **Diện tích thực:** {} cm²".format(area_cm2) if area_cm2 else "",
                "• **Tỷ lệ chiếm não:** {}".format(occupancy) if occupancy else "",
                "• **Vị trí u:** {}".format(location) if location else "",
                "" if (area_px or area_cm2) else "• Không phát hiện khối u đáng kể.",
                "\n**Phân loại lâm sàng:**",
                "• < 2 cm² → Nhỏ: theo dõi định kỳ",
                "• 2–10 cm² → Trung bình: cần đánh giá thêm",
                "• > 10 cm² → Lớn: thường cần can thiệp",
                "\n⚕️ Diện tích 2D chưa phản ánh đầy đủ thể tích 3D thực tế.",
            ]
            return build("📐 **Kích thước và diện tích khối u:**", [i for i in items if i is not None])

        # ── hình dạng / đặc trưng ──────────────────────────────────────
        shape_kws = ["hình dạng","đặc trưng","feature","irregularity","bất đối",
                     "compactness","boundary","đường biên","phức tạp","shape"]
        if any(k in q for k in shape_kws):
            items = [
                "• **Độ bất đối xứng (irregularity):** {}".format(irregularity) if irregularity else "",
                "• **Độ tròn (compactness):** {}".format(compactness) if compactness else "",
                "• **Phức tạp đường biên:** {}".format(boundary) if boundary else "",
                "• **Số vùng u:** {}".format(num_regions) if num_regions else "",
                "• **Vị trí:** {}".format(location) if location else "",
                "\n**Ý nghĩa lâm sàng:**",
                "• Irregularity > 0.7 → nguy cơ ác tính cao",
                "• Compactness gần 1.0 → hình tròn đều → thường lành tính hơn",
                "• Boundary complexity cao → đường biên gai góc → có thể xâm lấn",
            ]
            return build("🔬 **Đặc trưng hình dạng khối u:**", [i for i in items if i is not None])

        # ── midline shift ───────────────────────────────────────────────
        if any(k in q for k in ["midline","shift","dịch chuyển","đường giữa"]):
            status = midline if midline else "Không xác định"
            danger_note = ("\n🔴 Phát hiện midline shift — dấu hiệu nguy hiểm nghiêm trọng!\n"
                           "→ Cần theo dõi áp lực nội sọ và xem xét can thiệp khẩn cấp."
                           if midline == "Có"
                           else "\n🟢 Không phát hiện dịch chuyển đường giữa — dấu hiệu tích cực.")
            items = [
                "• Kết quả phân tích: **{}**".format(status),
                "\n**Midline Shift là gì?**",
                "Khi khối u hoặc phù não đẩy các cấu trúc não qua đường trung tâm.",
                danger_note,
            ]
            return build("⚠️ **Midline Shift (Dịch chuyển đường giữa não):**", items)

        # ── vị trí ─────────────────────────────────────────────────────
        if any(k in q for k in ["vị trí","location","nằm ở","ở đâu","thùy"]):
            items = [
                "• **Vị trí:** {}".format(location) if location else "",
                "• **Diện tích:** {} cm²".format(area_cm2) if area_cm2 else "",
                "• **Tỷ lệ chiếm não:** {}".format(occupancy) if occupancy else "",
                "\n**Ý nghĩa vị trí:**",
                "• Thùy trán: ảnh hưởng hành vi, ngôn ngữ, vận động",
                "• Thùy đỉnh/thái dương: ảnh hưởng cảm giác, trí nhớ",
                "• Thùy chẩm/tiểu não: ảnh hưởng thị giác, thăng bằng",
            ]
            return build("📍 **Vị trí khối u:**", [i for i in items if i is not None])

        # ── khuyến nghị ─────────────────────────────────────────────────
        if any(k in q for k in ["khuyến nghị","tiếp theo","bước tiếp","recommend","nên gặp"]):
            rec_list = [r.strip() for r in recommend.split("|") if r.strip()] if recommend else []
            items = ["• " + r for r in rec_list]
            if explanation:
                items.append("\n📊 **Giải thích:**\n{}".format(explanation))
            items.append("\n⚕️ {}".format(risk_advice(score_num)))
            return build("💡 **Khuyến nghị từ kết quả phân tích:**", items)

        # ── tóm tắt tổng quan ────────────────────────────────────────────
        items = [
            "• **Risk Score:** {}".format(risk_score) if risk_score else "",
            "• **Mức độ:** {}".format(risk_level) if risk_level else "",
            "• **Nghiêm trọng:** {}".format(severity) if severity else "",
            "• **Phát hiện u:** {}".format(detected) if detected else "",
            "• **Diện tích u:** {} cm²".format(area_cm2) if area_cm2 else "",
            "• **Tỷ lệ chiếm não:** {}".format(occupancy) if occupancy else "",
            "• **Vị trí:** {}".format(location) if location else "",
            "• **Midline Shift:** {}".format(midline) if midline else "",
            "\nBạn muốn tôi giải thích chi tiết về:\n• Mức độ nguy hiểm  • Diện tích / vị trí u\n• Hình dạng khối u  • Midline shift  • Khuyến nghị",
        ]
        return build("📋 **Tóm tắt kết quả phân tích BrainAI:**", [i for i in items if i is not None])

    # ── Không có context → knowledge base tĩnh ───────────────────────
    best_match = None
    best_score_kb = 0
    for entry in KNOWLEDGE_BASE:
        score = sum(1 for kw in entry["keywords"] if kw in q)
        if score > best_score_kb:
            best_score_kb = score
            best_match = entry

    if best_match and best_score_kb > 0:
        return best_match["answer"] + "\n\n*📌 Thông tin từ cơ sở kiến thức nội bộ BrainAI.*"

    return (
        "🤔 Câu hỏi của bạn nằm ngoài cơ sở kiến thức nội bộ hiện tại.\n\n"
        "**Bạn có thể hỏi về:**\n"
        "• Triệu chứng / mức độ nguy hiểm của khối u\n"
        "• Diện tích / vị trí khối u\n"
        "• Hình dạng và đặc trưng u\n"
        "• Midline shift là gì\n"
        "• Phương pháp điều trị\n"
        "• Lịch tái khám\n\n"
        "⚕️ Tải ảnh MRI lên để nhận kết quả phân tích với số liệu thực tế."
    )


async def call_anthropic_api(question, history, analysis_context=None):
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY not configured")

    system = SYSTEM_PROMPT
    if analysis_context:
        system += (
            "\n\n---\nKẾT QUẢ PHÂN TÍCH BRAINAI (dữ liệu thực từ hệ thống):\n"
            + analysis_context
            + "\n---\n"
            "Hãy ưu tiên sử dụng CHÍNH XÁC các số liệu này khi trả lời. "
            "Giải thích ý nghĩa lâm sàng của từng chỉ số. "
            "Đừng phán đoán hoặc suy diễn thêm ngoài dữ liệu đã cho."
        )

    messages = []
    for msg in history[-8:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})

    async with httpx.AsyncClient(timeout=25.0) as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 700,
                "system": system,
                "messages": messages,
            },
        )
        if response.status_code != 200:
            logger.error("Anthropic API error: {}".format(response.status_code))
            raise HTTPException(status_code=502, detail="AI service error")
        return response.json()["content"][0]["text"]


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
@router.post("/chat", response_model=ChatResponse)
async def medical_chat(request: ChatRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if len(question) > 1000:
        raise HTTPException(status_code=400, detail="Question too long")

    history = [{"role": m.role, "content": m.content} for m in (request.history or [])]
    ctx = request.analysis_context  # ← nhận context từ frontend

    try:
        answer = await call_anthropic_api(question, history, ctx)
        return ChatResponse(answer=answer)
    except ValueError:
        # Không có API key → dùng fallback thông minh với context
        return ChatResponse(answer=get_fallback_answer(question, ctx), model="local-kb")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return ChatResponse(answer=get_fallback_answer(question, ctx), model="local-kb")
