import { useEffect, useState } from 'react';

function RiskGauge({ score, color }) {
  const [displayed, setDisplayed] = useState(0);
  const R = 54;
  const CIRC = 2 * Math.PI * R;

  useEffect(() => {
    let start = null;
    const duration = 1200;
    const animate = (ts) => {
      if (!start) start = ts;
      const prog = Math.min((ts - start) / duration, 1);
      const ease = 1 - Math.pow(1 - prog, 3);
      setDisplayed(Math.round(score * ease));
      if (prog < 1) requestAnimationFrame(animate);
    };
    requestAnimationFrame(animate);
  }, [score]);

  const offset = CIRC * (1 - displayed / 100);

  return (
    <div className="risk-gauge">
      <svg width="140" height="140" viewBox="0 0 140 140">
        {/* Track */}
        <circle cx="70" cy="70" r={R} fill="none"
          stroke="rgba(0,200,255,0.07)" strokeWidth="10" />
        {/* Progress */}
        <circle cx="70" cy="70" r={R} fill="none"
          stroke={color} strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={CIRC}
          strokeDashoffset={offset}
          style={{ transform: 'rotate(-90deg)', transformOrigin: '70px 70px', transition: 'none', filter: `drop-shadow(0 0 8px ${color})` }}
        />
        {/* Tick marks */}
        {[0, 25, 50, 75].map(pct => {
          const angle = (pct / 100) * 360 - 90;
          const rad = (angle * Math.PI) / 180;
          const x1 = 70 + (R - 6) * Math.cos(rad);
          const y1 = 70 + (R - 6) * Math.sin(rad);
          const x2 = 70 + (R + 6) * Math.cos(rad);
          const y2 = 70 + (R + 6) * Math.sin(rad);
          return <line key={pct} x1={x1} y1={y1} x2={x2} y2={y2} stroke="rgba(0,200,255,0.2)" strokeWidth="1.5" />;
        })}
      </svg>
      <div className="risk-number">
        <span style={{ color }}>{displayed}</span>
        <small>/100</small>
      </div>
    </div>
  );
}

export default function RiskPanel({ risk, features }) {
  const color = risk.risk_color;

  const levelEmoji = {
    'Rất cao': '🔴',
    'Cao': '🟠',
    'Trung bình': '🟡',
    'Thấp': '🟢',
  }[risk.risk_level] || '⚪';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>

      {/* RISK SCORE CARD */}
      <div className="card risk-card animate-in" style={{ '--risk-color': color }}>
        <div className="card-header">
          <span style={{ fontSize: 16 }}>⚕️</span>
          <span className="card-title">Đánh giá nguy cơ lâm sàng</span>
        </div>

        <div className="risk-score-display">
          <RiskGauge score={risk.risk_score} color={color} />

          <div
            className="risk-level-badge"
            style={{
              background: `${color}18`,
              border: `1px solid ${color}40`,
              color: color,
              marginTop: 4,
            }}
          >
            {levelEmoji} Nguy cơ {risk.risk_level}
          </div>
          <div className="risk-severity" style={{ marginTop: 6 }}>
            Mức độ: <span style={{ color }}>{risk.severity}</span>
          </div>
        </div>
      </div>

      {/* FEATURES CARD */}
      <div className="card animate-in">
        <div className="card-header">
          <span style={{ fontSize: 16 }}>📐</span>
          <span className="card-title">Đặc trưng khối u</span>
          {features.tumor_detected
            ? <span className="card-tag" style={{ color: '#ff6b35', borderColor: '#ff6b3530', background: '#ff6b3510' }}>DETECTED</span>
            : <span className="card-tag" style={{ color: 'var(--green)', borderColor: 'rgba(0,255,136,0.2)', background: 'var(--green-dim)' }}>CLEAR</span>}
        </div>

        {features.tumor_detected ? (
          <div className="features-grid">
            <div className="feat-item">
              <div className="feat-label">Diện tích</div>
              <div className="feat-value highlight">
                {features.tumor_area_cm2}
                <span className="feat-unit">cm²</span>
              </div>
            </div>
            <div className="feat-item">
              <div className="feat-label">Tỷ lệ chiếm</div>
              <div className="feat-value highlight">
                {features.occupancy_ratio}
                <span className="feat-unit">%</span>
              </div>
            </div>
            <div className="feat-item">
              <div className="feat-label">Bất đối xứng</div>
              <div className="feat-value">
                {features.shape_irregularity.toFixed(2)}
              </div>
            </div>
            <div className="feat-item">
              <div className="feat-label">Độ compact</div>
              <div className="feat-value">
                {features.compactness.toFixed(2)}
              </div>
            </div>
            <div className="feat-item">
              <div className="feat-label">Biên phức tạp</div>
              <div className="feat-value">
                {features.boundary_complexity.toFixed(1)}
              </div>
            </div>
            <div className="feat-item">
              <div className="feat-label">Số vùng</div>
              <div className="feat-value">
                {features.num_regions}
              </div>
            </div>
            <div className="feat-item" style={{ gridColumn: 'span 2' }}>
              <div className="feat-label">Midline Shift</div>
              <div className="feat-value" style={{ fontSize: 14, color: features.midline_shift ? '#ff6b35' : 'var(--green)' }}>
                {features.midline_shift ? '⚠️ Có dịch chuyển đường giữa' : '✓ Không có dịch chuyển'}
              </div>
            </div>
            <div className="feat-item feat-location" style={{ gridColumn: 'span 2' }}>
              <div className="feat-label">Vị trí khối u</div>
              <div className="feat-value" style={{ fontSize: 13 }}>📍 {features.location}</div>
            </div>
          </div>
        ) : (
          <div style={{ padding: '24px', textAlign: 'center', color: 'var(--green)', fontFamily: 'var(--font-mono)', fontSize: 13 }}>
            ✓ Không phát hiện khối u
          </div>
        )}
      </div>

      {/* FIRED RULES */}
      <div className="card animate-in">
        <div className="card-header">
          <span style={{ fontSize: 16 }}>⚖️</span>
          <span className="card-title">Luật y khoa kích hoạt</span>
          <span className="card-tag">{risk.fired_rules.length} luật</span>
        </div>
        <div className="rules-list">
          {risk.fired_rules.map((rule, i) => (
            <div key={i} className="rule-item">
              <span className="rule-icon">
                {rule.includes('cao') || rule.includes('phức tạp') || rule.includes('bất đối') || rule.includes('dịch chuyển') || rule.includes('di căn') || rule.includes('lớn')
                  ? '⚠️' : '✓'}
              </span>
              <span>{rule}</span>
            </div>
          ))}
        </div>
      </div>

      {/* RECOMMENDATIONS */}
      <div className="card animate-in">
        <div className="card-header">
          <span style={{ fontSize: 16 }}>💊</span>
          <span className="card-title">Khuyến nghị lâm sàng</span>
        </div>
        <div className="rec-list">
          {risk.recommendations.map((rec, i) => (
            <div key={i} className={`rec-item${rec.includes('KHẨN') || rec.includes('khẩn') ? ' urgent' : ''}`}>
              <span className="rec-icon">
                {rec.includes('KHẨN') || rec.includes('khẩn') ? '🚨' : '→'}
              </span>
              <span>{rec}</span>
            </div>
          ))}
        </div>
      </div>

      {/* XAI EXPLANATION */}
      <div className="card animate-in">
        <div className="card-header">
          <span style={{ fontSize: 16 }}>🔍</span>
          <span className="card-title">Giải thích AI</span>
          <span className="card-tag">XAI</span>
        </div>
        <div className="explanation-box">
          {risk.explanation}
        </div>
        <div style={{
          margin: '0 24px 20px',
          padding: '12px 16px',
          borderRadius: 8,
          background: 'rgba(255, 214, 10, 0.05)',
          border: '1px solid rgba(255, 214, 10, 0.15)',
          fontFamily: 'var(--font-mono)',
          fontSize: 11,
          color: 'rgba(255,214,10,0.7)',
          lineHeight: 1.6,
        }}>
          ⚠️ Kết quả từ AI chỉ mang tính tham khảo. Cần có sự xác nhận của bác sĩ chuyên khoa thần kinh trước khi đưa ra quyết định lâm sàng.
        </div>
      </div>

    </div>
  );
}
