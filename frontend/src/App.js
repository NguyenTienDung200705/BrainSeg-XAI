import { useState, useCallback } from 'react';
import './index.css';
import NeuralCanvas from './components/NeuralCanvas';
import Header from './components/Header';
import UploadZone from './components/UploadZone';
import ProcessingPanel from './components/ProcessingPanel';
import ImageViewer from './components/ImageViewer';
import RiskPanel from './components/RiskPanel';
import MedicalChatbot from './components/MedicalChatbot';
import ChatToggleButton from './components/ChatToggleButton';
import { predictTumor } from './services/api';

// Simulated progress while waiting for response
function useSimProgress(active) {
  const [prog, setProg] = useState(0);

  const start = useCallback(() => {
    setProg(0);
    let val = 0;
    const STAGES = [
      { target: 18, speed: 120 },
      { target: 40, speed: 90 },
      { target: 62, speed: 110 },
      { target: 80, speed: 130 },
      { target: 92, speed: 200 },
      { target: 97, speed: 350 },
    ];
    let stageIdx = 0;
    const tick = setInterval(() => {
      if (stageIdx >= STAGES.length) { clearInterval(tick); return; }
      const { target, speed } = STAGES[stageIdx];
      val = Math.min(val + 1, target);
      setProg(val);
      if (val >= target) stageIdx++;
    }, 80);
    return () => clearInterval(tick);
  }, []);

  const finish = useCallback(() => setProg(100), []);
  const reset = useCallback(() => setProg(0), []);

  return { prog, start, finish, reset };
}

export default function App() {
  const [phase, setPhase] = useState('idle');
  const [chatOpen, setChatOpen] = useState(false); // idle | processing | result | error
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [previewSrc, setPreviewSrc] = useState(null);
  const { prog, start, finish, reset } = useSimProgress();

  const handleFile = useCallback(async (file) => {
    setPhase('processing');
    setError('');
    setResult(null);

    // Show preview of uploaded file
    const reader = new FileReader();
    reader.onload = e => setPreviewSrc(e.target.result);
    reader.readAsDataURL(file);

    const cleanup = start();

    try {
      const data = await predictTumor(file);
      finish();
      setTimeout(() => {
        setResult(data);
        setPhase('result');
        setChatOpen(true); // auto-mở chatbot khi có kết quả
      }, 400);
    } catch (err) {
      cleanup?.();
      reset();
      const msg = err?.response?.data?.detail || err.message || 'Lỗi không xác định';
      setError(msg);
      setPhase('error');
    }
  }, [start, finish, reset]);

  const handleReset = () => {
    setPhase('idle');
    setResult(null);
    setError('');
    setPreviewSrc(null);
    reset();
  };

  return (
    <div className="app-layout">
      <NeuralCanvas />
      <Header />

      <main className="main-content">

        {/* ===== IDLE: Hero + Upload ===== */}
        {phase === 'idle' && (
          <>
            <div className="hero">
              <div className="hero-tag">Explainable Medical AI · U-Net · Grad-CAM</div>
              <h1 className="hero-title">
                Phân tích khối u não<br />thông minh & minh bạch
              </h1>
              <p className="hero-sub">
                Hệ thống AI tích hợp U-Net segmentation, Medical Rule Engine
                và Explainable AI để hỗ trợ chẩn đoán lâm sàng từ ảnh MRI não.
              </p>
            </div>

            <div style={{ maxWidth: 780, margin: '0 auto' }}>
              <UploadZone onFile={handleFile} />

              {/* Stats row */}
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(4, 1fr)',
                gap: 16,
                marginTop: 32,
              }}>
                {[
                  { num: 'U-Net', label: 'Kiến trúc mô hình', icon: '🧠' },
                  { num: 'Grad-CAM', label: 'XAI Visualization', icon: '🔍' },
                  { num: '6 bước', label: 'Pipeline phân tích', icon: '⚙️' },
                  { num: 'Tiếng Việt', label: 'Giao diện & báo cáo', icon: '🇻🇳' },
                ].map(s => (
                  <div key={s.label} style={{
                    background: 'rgba(10,21,32,0.6)',
                    border: '1px solid var(--border-dim)',
                    borderRadius: 12,
                    padding: '16px 20px',
                    textAlign: 'center',
                  }}>
                    <div style={{ fontSize: 24, marginBottom: 6 }}>{s.icon}</div>
                    <div style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: 14, color: 'var(--cyan)', marginBottom: 4 }}>{s.num}</div>
                    <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-secondary)' }}>{s.label}</div>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}

        {/* ===== PROCESSING ===== */}
        {phase === 'processing' && (
          <div style={{ maxWidth: 600, margin: '60px auto' }}>
            <div className="card">
              <div className="card-header">
                <span style={{ fontSize: 16 }}>⚙️</span>
                <span className="card-title">Đang phân tích...</span>
              </div>
              {previewSrc && (
                <div style={{ padding: '16px 24px 0', display: 'flex', justifyContent: 'center' }}>
                  <img src={previewSrc} alt="preview" style={{
                    height: 120,
                    borderRadius: 8,
                    border: '1px solid var(--border-dim)',
                    objectFit: 'contain',
                    background: '#000',
                  }} />
                </div>
              )}
              <ProcessingPanel progress={prog} />
            </div>
          </div>
        )}

        {/* ===== ERROR ===== */}
        {phase === 'error' && (
          <div style={{ maxWidth: 600, margin: '60px auto', textAlign: 'center' }}>
            <div className="card" style={{ padding: '40px 32px' }}>
              <div style={{ fontSize: 48, marginBottom: 16 }}>⚠️</div>
              <div style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: 20, marginBottom: 12, color: 'var(--red-hot)' }}>
                Đã xảy ra lỗi
              </div>
              <div style={{
                fontFamily: 'var(--font-mono)',
                fontSize: 12,
                color: 'var(--text-secondary)',
                background: 'var(--bg-panel)',
                border: '1px solid var(--border-dim)',
                borderRadius: 8,
                padding: '12px 16px',
                marginBottom: 24,
                textAlign: 'left',
                wordBreak: 'break-word',
              }}>
                {error}
              </div>
              <button className="btn-reset" onClick={handleReset} style={{ margin: '0 auto' }}>
                ← Thử lại
              </button>
            </div>
          </div>
        )}

        {/* ===== RESULT ===== */}
        {phase === 'result' && result && (
          <>
            <div className="results-header">
              <div>
                <div className="results-title">Kết quả <span>phân tích</span></div>
                <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-secondary)', marginTop: 4 }}>
                  Phân tích hoàn tất · {new Date().toLocaleString('vi-VN')}
                </div>
              </div>
              <button className="btn-reset" onClick={handleReset}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polyline points="1 4 1 10 7 10"/>
                  <path d="M3.51 15a9 9 0 1 0 .49-4.76"/>
                </svg>
                Phân tích ảnh mới
              </button>
            </div>

            <div className="results-layout">
              {/* Left: images */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
                <ImageViewer result={result} />

                {/* Pipeline info */}
                <div className="card animate-in">
                  <div className="card-header">
                    <span style={{ fontSize: 16 }}>🔬</span>
                    <span className="card-title">Pipeline xử lý</span>
                    <span className="card-tag" style={{ color: 'var(--green)', borderColor: 'rgba(0,255,136,0.2)', background: 'var(--green-dim)' }}>COMPLETED</span>
                  </div>
                  <div style={{ padding: '16px 24px 20px', display: 'flex', gap: 0, flexWrap: 'nowrap', overflowX: 'auto' }}>
                    {[
                      { label: 'Tiền xử lý', icon: '⚙️', done: true },
                      { label: 'U-Net', icon: '🧠', done: true },
                      { label: 'Feature Ext.', icon: '📐', done: true },
                      { label: 'Rule Engine', icon: '⚖️', done: true },
                      { label: 'Grad-CAM', icon: '🔍', done: true },
                      { label: 'Báo cáo', icon: '📋', done: true },
                    ].map((step, i, arr) => (
                      <div key={step.label} style={{ display: 'flex', alignItems: 'center', gap: 0 }}>
                        <div style={{
                          display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6, minWidth: 70,
                        }}>
                          <div style={{
                            width: 40, height: 40, borderRadius: '50%',
                            background: 'var(--green-dim)',
                            border: '1px solid rgba(0,255,136,0.3)',
                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                            fontSize: 18,
                            boxShadow: '0 0 10px rgba(0,255,136,0.1)',
                          }}>{step.icon}</div>
                          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--green)', textAlign: 'center' }}>{step.label}</span>
                        </div>
                        {i < arr.length - 1 && (
                          <div style={{
                            width: 24, height: 1,
                            background: 'linear-gradient(90deg, rgba(0,255,136,0.4), rgba(0,200,255,0.4))',
                            marginBottom: 20,
                            flexShrink: 0,
                          }} />
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Right: analysis */}
              <RiskPanel risk={result.risk} features={result.features} onAskChatbot={(q) => { setChatOpen(true); window.__chatbotAsk && window.__chatbotAsk(q); }} />
            </div>
          </>
        )}

      </main>

      <footer className="footer">
        <div className="footer-text">
          © 2025 BrainAI · Explainable Medical AI System
        </div>
        <div className="footer-warn">
          ⚠️ Chỉ dùng cho mục đích nghiên cứu & học thuật. Không thay thế chẩn đoán y khoa chính thức.
        </div>
      </footer>

      {/* ===== CHATBOT ===== */}
      {!chatOpen && (
        <ChatToggleButton
          isOpen={false}
          onClick={() => setChatOpen(true)}
        />
      )}
      <MedicalChatbot
        isOpen={chatOpen}
        onToggle={() => setChatOpen(c => !c)}
        analysisResult={result}
      />
    </div>
  );
}