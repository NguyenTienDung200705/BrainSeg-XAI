import { useEffect, useState } from 'react';

const STAGES = [
  { id: 'preprocess', label: 'Tiền xử lý ảnh MRI' },
  { id: 'inference', label: 'U-Net Segmentation' },
  { id: 'features', label: 'Trích xuất đặc trưng' },
  { id: 'rules', label: 'Medical Rule Engine' },
  { id: 'xai', label: 'Grad-CAM XAI' },
  { id: 'report', label: 'Tổng hợp báo cáo' },
];

export default function ProcessingPanel({ progress }) {
  const [activeIdx, setActiveIdx] = useState(0);

  useEffect(() => {
    const total = STAGES.length;
    const idx = Math.min(Math.floor((progress / 100) * total), total - 1);
    setActiveIdx(idx);
  }, [progress]);

  return (
    <div className="processing-panel">
      {/* Animated ring */}
      <div className="processing-ring">
        <svg width="100" height="100" viewBox="0 0 100 100">
          <circle cx="50" cy="50" r="42" fill="none" stroke="rgba(0,200,255,0.08)" strokeWidth="6" />
          <circle
            cx="50" cy="50" r="42"
            fill="none"
            stroke="url(#grad)"
            strokeWidth="6"
            strokeLinecap="round"
            strokeDasharray={`${2 * Math.PI * 42}`}
            strokeDashoffset={`${2 * Math.PI * 42 * (1 - progress / 100)}`}
            style={{ transition: 'stroke-dashoffset 0.4s ease' }}
          />
          <defs>
            <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#00c8ff" />
              <stop offset="100%" stopColor="#0066ff" />
            </linearGradient>
          </defs>
          {/* Brain icon inside */}
          <text x="50" y="57" textAnchor="middle" fontSize="26" fill="rgba(0,200,255,0.7)">🧠</text>
        </svg>
      </div>

      <div style={{ textAlign: 'center' }}>
        <div style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: 18, marginBottom: 4 }}>
          Đang phân tích ảnh MRI
        </div>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-secondary)' }}>
          {Math.round(progress)}% hoàn thành
        </div>
      </div>

      <div className="processing-stages">
        {STAGES.map((s, i) => {
          const state = i < activeIdx ? 'done' : i === activeIdx ? 'active' : 'pending';
          return (
            <div key={s.id} className={`stage-item ${state}`}>
              <div className="stage-dot" />
              <span>{s.label}</span>
              {state === 'done' && (
                <span style={{ marginLeft: 'auto', fontSize: 12 }}>✓</span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
