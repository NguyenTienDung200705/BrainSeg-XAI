import { useState } from 'react';

const VIEWS = [
  { key: 'original_b64', label: 'Ảnh Gốc', tag: 'MRI RAW' },
  { key: 'overlay_b64', label: 'Mask Overlay', tag: 'SEGMENTATION' },
  { key: 'mask_b64', label: 'Segmentation', tag: 'BINARY MASK' },
  { key: 'cam_overlay_b64', label: 'XAI Overlay', tag: 'GRAD-CAM' },
  { key: 'heatmap_b64', label: 'Heatmap', tag: 'ATTENTION MAP' },
];

export default function ImageViewer({ result }) {
  const [active, setActive] = useState(0);

  const current = VIEWS[active];
  const src = `data:image/png;base64,${result[current.key]}`;

  return (
    <div className="card animate-in">
      <div className="card-header">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--cyan)" strokeWidth="2">
          <rect x="3" y="3" width="18" height="18" rx="2"/>
          <circle cx="8.5" cy="8.5" r="1.5"/>
          <polyline points="21 15 16 10 5 21"/>
        </svg>
        <span className="card-title">Trực quan hoá kết quả</span>
        <span className="card-tag">{current.tag}</span>
      </div>

      {/* Tab selector */}
      <div className="image-tabs">
        {VIEWS.map((v, i) => (
          <button
            key={v.key}
            className={`img-tab${active === i ? ' active' : ''}`}
            onClick={() => setActive(i)}
          >
            {v.label}
          </button>
        ))}
      </div>

      {/* Main image + thumbnails */}
      <div className="image-display">
        <div className="img-main-wrapper">
          <img className="img-main" src={src} alt={current.label} />
          <div className="img-label">{current.label}</div>
        </div>

        <div className="img-thumbs">
          {VIEWS.map((v, i) => (
            <div key={v.key}>
              <div
                className={`img-thumb${active === i ? ' active' : ''}`}
                onClick={() => setActive(i)}
              >
                <img
                  src={`data:image/png;base64,${result[v.key]}`}
                  alt={v.label}
                />
              </div>
              <div className="thumb-label">{v.label.split(' ')[0]}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Compare strip */}
      <div style={{
        padding: '0 24px 20px',
        display: 'grid',
        gridTemplateColumns: 'repeat(5, 1fr)',
        gap: 8,
      }}>
        {VIEWS.map((v, i) => (
          <div
            key={v.key}
            onClick={() => setActive(i)}
            style={{
              cursor: 'pointer',
              borderRadius: 6,
              overflow: 'hidden',
              border: `1px solid ${active === i ? 'var(--cyan)' : 'var(--border-dim)'}`,
              aspectRatio: '1',
              background: '#000',
              boxShadow: active === i ? '0 0 10px rgba(0,200,255,0.2)' : 'none',
              transition: 'all 0.2s',
            }}
          >
            <img
              src={`data:image/png;base64,${result[v.key]}`}
              alt={v.label}
              style={{ width: '100%', height: '100%', objectFit: 'contain', display: 'block' }}
            />
          </div>
        ))}
      </div>
    </div>
  );
}
