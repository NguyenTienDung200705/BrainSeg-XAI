import { useRef, useState, useCallback } from 'react';

const ACCEPTED = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp', 'image/tiff', 'image/webp'];

export default function UploadZone({ onFile }) {
  const inputRef = useRef(null);
  const [drag, setDrag] = useState(false);

  const handleFile = useCallback((file) => {
    if (!file) return;
    if (!ACCEPTED.includes(file.type)) {
      alert('Chỉ chấp nhận file ảnh: PNG, JPG, BMP, TIFF, WEBP');
      return;
    }
    onFile(file);
  }, [onFile]);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDrag(false);
    const file = e.dataTransfer.files[0];
    handleFile(file);
  }, [handleFile]);

  const onDragOver = (e) => { e.preventDefault(); setDrag(true); };
  const onDragLeave = () => setDrag(false);

  return (
    <div
      className={`upload-zone${drag ? ' drag-active' : ''}`}
      onClick={() => inputRef.current?.click()}
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED.join(',')}
        style={{ display: 'none' }}
        onChange={e => handleFile(e.target.files[0])}
      />

      <div className="upload-icon">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <polyline points="17 8 12 3 7 8"/>
          <line x1="12" y1="3" x2="12" y2="15"/>
        </svg>
      </div>

      <div className="upload-title">
        {drag ? 'Thả ảnh MRI vào đây' : 'Tải ảnh MRI não lên'}
      </div>
      <div className="upload-desc" style={{ marginTop: 8 }}>
        Kéo thả file hoặc click để chọn từ máy tính
      </div>

      <div className="upload-formats">
        {['PNG', 'JPG', 'BMP', 'TIFF', 'WEBP'].map(f => (
          <span key={f} className="format-badge">{f}</span>
        ))}
        <span className="format-badge">Tối đa 20MB</span>
      </div>

      <div style={{
        marginTop: 32,
        display: 'grid',
        gridTemplateColumns: 'repeat(3, 1fr)',
        gap: 12,
        maxWidth: 480,
        marginLeft: 'auto',
        marginRight: 'auto',
      }}>
        {[
          { icon: '🧠', label: 'Phân đoạn U-Net', desc: 'Xác định vùng khối u tự động' },
          { icon: '📊', label: 'Đánh giá nguy cơ', desc: 'Luật y khoa chuyên sâu' },
          { icon: '🔍', label: 'Grad-CAM XAI', desc: 'Giải thích quyết định AI' },
        ].map(item => (
          <div key={item.label} style={{
            background: 'rgba(10, 21, 32, 0.6)',
            border: '1px solid rgba(0, 200, 255, 0.08)',
            borderRadius: 10,
            padding: '14px 12px',
            textAlign: 'center',
          }}>
            <div style={{ fontSize: 22, marginBottom: 6 }}>{item.icon}</div>
            <div style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: 12, marginBottom: 4 }}>{item.label}</div>
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-secondary)', lineHeight: 1.5 }}>{item.desc}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
