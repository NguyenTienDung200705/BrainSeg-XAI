"""
Medical Rule Engine — đánh giá mức độ nguy cơ dựa trên đặc trưng khối u.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class RuleResult:
    risk_level: str          # "Thấp" | "Trung bình" | "Cao" | "Rất cao"
    risk_score: int          # 0-100
    severity: str            # "Bình thường" | "Nhẹ" | "Trung bình" | "Nghiêm trọng" | "Nguy kịch"
    risk_color: str          # hex color for UI
    fired_rules: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    explanation: str = ""


def evaluate_risk(features: dict) -> RuleResult:
    if not features.get("tumor_detected", False):
        return RuleResult(
            risk_level="Thấp",
            risk_score=0,
            severity="Bình thường",
            risk_color="#00ff88",
            fired_rules=["Không phát hiện khối u trong ảnh MRI"],
            recommendations=["Tiếp tục theo dõi định kỳ", "Chụp MRI kiểm tra sau 12 tháng"],
            explanation="Không phát hiện bất kỳ vùng bất thường nào trên ảnh MRI não.",
        )

    score = 0
    fired = []
    recs = []

    occ = features["occupancy_ratio"]
    irr = features["shape_irregularity"]
    bnd = features["boundary_complexity"]
    comp = features["compactness"]
    mid = features["midline_shift"]
    n_reg = features["num_regions"]
    area_cm2 = features["tumor_area_cm2"]

    # --- Rule 1: Occupancy ---
    if occ > 20:
        score += 30
        fired.append(f"Tỷ lệ chiếm vùng não rất cao ({occ}% > 20%)")
        recs.append("Cần can thiệp phẫu thuật khẩn cấp")
    elif occ > 10:
        score += 20
        fired.append(f"Tỷ lệ chiếm vùng não cao ({occ}% > 10%)")
        recs.append("Hội chẩn chuyên khoa thần kinh ngay")
    elif occ > 5:
        score += 10
        fired.append(f"Tỷ lệ chiếm vùng não ở mức trung bình ({occ}%)")
        recs.append("Theo dõi sát và chụp MRI định kỳ 3 tháng/lần")
    else:
        fired.append(f"Tỷ lệ chiếm vùng não thấp ({occ}%)")

    # --- Rule 2: Shape Irregularity ---
    if irr > 0.7:
        score += 25
        fired.append(f"Hình dạng khối u rất bất đối xứng (irregularity={irr:.2f})")
        recs.append("Sinh thiết để xác định tính ác tính")
    elif irr > 0.5:
        score += 15
        fired.append(f"Hình dạng khối u bất đối xứng vừa (irregularity={irr:.2f})")
    else:
        fired.append(f"Hình dạng khối u tương đối đều đặn (irregularity={irr:.2f})")

    # --- Rule 3: Boundary Complexity ---
    if bnd > 15:
        score += 20
        fired.append(f"Đường biên khối u rất phức tạp (complexity={bnd:.1f})")
        recs.append("Chỉ định PET-CT để đánh giá độ xâm lấn")
    elif bnd > 10:
        score += 10
        fired.append(f"Đường biên khối u phức tạp vừa (complexity={bnd:.1f})")
    else:
        fired.append(f"Đường biên khối u tương đối đơn giản (complexity={bnd:.1f})")

    # --- Rule 4: Midline Shift ---
    if mid:
        score += 20
        fired.append("Có dấu hiệu dịch chuyển đường giữa não (Midline Shift)")
        recs.append("Theo dõi áp lực nội sọ, xem xét can thiệp giải áp")

    # --- Rule 5: Multiple regions ---
    if n_reg > 3:
        score += 15
        fired.append(f"Phát hiện nhiều vùng khối u ({n_reg} vùng) — nghi ngờ di căn")
        recs.append("Chụp MRI toàn thân, tìm kiếm ổ di căn nguyên phát")
    elif n_reg > 1:
        score += 8
        fired.append(f"Phát hiện {n_reg} vùng khối u riêng biệt")

    # --- Rule 6: Absolute size ---
    if area_cm2 > 10:
        score += 15
        fired.append(f"Diện tích khối u rất lớn ({area_cm2} cm²)")
    elif area_cm2 > 4:
        score += 8
        fired.append(f"Diện tích khối u trung bình ({area_cm2} cm²)")
    else:
        fired.append(f"Diện tích khối u nhỏ ({area_cm2} cm²)")

    # Clamp
    score = min(score, 100)

    # Risk mapping
    if score >= 70:
        risk_level = "Rất cao"
        severity = "Nguy kịch"
        risk_color = "#ff2d55"
        recs.insert(0, "⚠️ KHẨN CẤP: Cần nhập viện và can thiệp ngay lập tức")
    elif score >= 50:
        risk_level = "Cao"
        severity = "Nghiêm trọng"
        risk_color = "#ff6b35"
        recs.insert(0, "Cần điều trị chuyên khoa trong vòng 48 giờ")
    elif score >= 30:
        risk_level = "Trung bình"
        severity = "Trung bình"
        risk_color = "#ffd60a"
        if not recs:
            recs.append("Theo dõi và điều trị theo phác đồ của bác sĩ")
    else:
        risk_level = "Thấp"
        severity = "Nhẹ"
        risk_color = "#00c896"
        if not recs:
            recs.append("Theo dõi định kỳ 6 tháng/lần")

    explanation = (
        f"Hệ thống phân tích {len(fired)} đặc trưng y khoa và kích hoạt "
        f"{sum(1 for r in fired if any(k in r for k in ['cao', 'phức tạp', 'bất đối', 'dịch chuyển', 'di căn', 'lớn']))} "
        f"cờ nguy cơ. Điểm nguy cơ tổng hợp: {score}/100. "
        f"Phân loại: {severity}."
    )

    return RuleResult(
        risk_level=risk_level,
        risk_score=score,
        severity=severity,
        risk_color=risk_color,
        fired_rules=fired,
        recommendations=list(dict.fromkeys(recs)),  # deduplicate
        explanation=explanation,
    )
