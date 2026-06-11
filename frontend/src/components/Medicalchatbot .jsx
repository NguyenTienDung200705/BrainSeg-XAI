import { useState, useRef, useEffect, useCallback } from 'react';
import { sendMedicalQuestion } from '../services/api';

// ── Quick question sets ──────────────────────────────────────────────────────
const QUICK_DEFAULT = [
  'Khối u não là gì?',
  'Triệu chứng của u não?',
  'U lành và u ác tính khác nhau thế nào?',
  'Grad-CAM hoạt động như thế nào?',
  'Khi nào cần gặp bác sĩ ngay?',
];
const QUICK_WITH_RESULT = [
  'Khối u này có nguy hiểm không?',
  'Diện tích khối u là bao nhiêu?',
  'Giải thích risk score cho tôi',
  'Phân tích đặc điểm hình dạng khối u',
  'Tôi cần làm gì tiếp theo?',
  'Có dấu hiệu midline shift không?',
];

// ── Build context string from analysisResult ─────────────────────────────────
function buildContext(result) {
  if (!result) return null;
  try {
    const r = result.risk     || {};
    const f = result.features || {};
    const lines = ['=== KẾT QUẢ PHÂN TÍCH BRAINAI ==='];

    // risk fields (match exact keys from inference.py)
    if (r.risk_score      != null) lines.push(`Risk Score: ${r.risk_score}/100`);
    if (r.risk_level)              lines.push(`Mức độ: ${r.risk_level}`);
    if (r.severity)                lines.push(`Mức độ nghiêm trọng: ${r.severity}`);
    if (r.explanation)             lines.push(`Giải thích: ${r.explanation}`);
    if (r.fired_rules?.length)     lines.push(`Quy tắc kích hoạt: ${r.fired_rules.join(' | ')}`);
    if (r.recommendations?.length) lines.push(`Khuyến nghị: ${r.recommendations.join(' | ')}`);

    // feature fields (match exact keys from feature_extraction.py)
    lines.push('--- Đặc trưng khối u ---');
    if (f.tumor_detected != null)   lines.push(`Phát hiện u: ${f.tumor_detected ? 'Có' : 'Không'}`);
    if (f.tumor_area_px  != null)   lines.push(`Diện tích u: ${f.tumor_area_px} pixels`);
    if (f.tumor_area_cm2 != null)   lines.push(`Diện tích u (cm²): ${f.tumor_area_cm2} cm²`);
    if (f.occupancy_ratio != null)  lines.push(`Tỷ lệ chiếm não: ${f.occupancy_ratio}%`);
    if (f.num_regions     != null)  lines.push(`Số vùng u: ${f.num_regions}`);
    if (f.shape_irregularity != null) lines.push(`Độ bất đối xứng: ${f.shape_irregularity}`);
    if (f.compactness     != null)  lines.push(`Độ tròn (compactness): ${f.compactness}`);
    if (f.boundary_complexity != null) lines.push(`Phức tạp đường biên: ${f.boundary_complexity}`);
    if (f.midline_shift   != null)  lines.push(`Midline Shift: ${f.midline_shift ? 'Có' : 'Không'}`);
    if (f.location)                 lines.push(`Vị trí u: ${f.location}`);
    if (f.centroid_x != null && f.centroid_y != null)
      lines.push(`Tâm u: (${f.centroid_x}, ${f.centroid_y})`);

    return lines.join('\n');
  } catch {
    return null;
  }
}

// ── Simple markdown renderer ─────────────────────────────────────────────────
function renderMarkdown(text) {
  if (!text) return [];
  const lines = text.split('\n');
  const elements = [];
  let key = 0;

  for (const line of lines) {
    // Bold **text**
    const parseBold = (str) => {
      const parts = str.split(/\*\*(.*?)\*\*/g);
      return parts.map((p, i) =>
        i % 2 === 1
          ? <strong key={i} style={{ color: 'var(--cyan)', fontWeight: 700 }}>{p}</strong>
          : p
      );
    };

    if (line.trim() === '') {
      elements.push(<div key={key++} style={{ height: 6 }} />);
    } else if (line.startsWith('• ') || line.startsWith('- ')) {
      elements.push(
        <div key={key++} style={{ display: 'flex', gap: 6, marginBottom: 3 }}>
          <span style={{ color: 'var(--cyan)', flexShrink: 0, marginTop: 1 }}>•</span>
          <span>{parseBold(line.slice(2))}</span>
        </div>
      );
    } else if (/^\d+\./.test(line)) {
      const num = line.match(/^(\d+\.)/)[1];
      elements.push(
        <div key={key++} style={{ display: 'flex', gap: 6, marginBottom: 3 }}>
          <span style={{ color: 'var(--cyan)', flexShrink: 0, fontFamily: 'var(--font-mono)', fontSize: 11 }}>{num}</span>
          <span>{parseBold(line.slice(num.length).trim())}</span>
        </div>
      );
    } else if (line.startsWith('---') || line.startsWith('===')) {
      elements.push(
        <div key={key++} style={{
          borderTop: '1px solid rgba(0,200,255,0.12)',
          margin: '6px 0',
        }} />
      );
    } else if (line.startsWith('# ')) {
      elements.push(
        <div key={key++} style={{
          fontWeight: 700, fontSize: 14,
          color: 'var(--cyan)', marginBottom: 4, marginTop: 4,
        }}>{parseBold(line.slice(2))}</div>
      );
    } else {
      elements.push(
        <div key={key++} style={{ marginBottom: 2 }}>
          {parseBold(line)}
        </div>
      );
    }
  }
  return elements;
}

// ── Sub-components ────────────────────────────────────────────────────────────
function TypingDots() {
  return (
    <div style={{ display: 'flex', gap: 4, alignItems: 'center', padding: '8px 0 4px' }}>
      {[0, 1, 2].map(i => (
        <div key={i} style={{
          width: 6, height: 6, borderRadius: '50%',
          background: 'var(--cyan)', opacity: 0.7,
          animation: `typing-bounce 1.2s ease-in-out ${i * 0.2}s infinite`,
        }} />
      ))}
    </div>
  );
}

function ChatMessage({ msg }) {
  const isUser = msg.role === 'user';
  return (
    <div style={{
      display: 'flex',
      justifyContent: isUser ? 'flex-end' : 'flex-start',
      marginBottom: 14,
      animation: 'msg-in 0.3s cubic-bezier(0.34,1.56,0.64,1)',
    }}>
      {!isUser && (
        <div style={{
          width: 28, height: 28, borderRadius: '50%',
          background: 'linear-gradient(135deg,rgba(0,200,255,.2),rgba(0,255,136,.1))',
          border: '1px solid rgba(0,200,255,.3)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 13, flexShrink: 0, marginRight: 8, marginTop: 2,
        }}>🧠</div>
      )}
      <div style={{
        maxWidth: '80%',
        padding: '10px 14px',
        borderRadius: isUser ? '16px 16px 4px 16px' : '4px 16px 16px 16px',
        background: isUser
          ? 'linear-gradient(135deg,rgba(0,200,255,.15),rgba(0,150,200,.08))'
          : 'rgba(8,18,28,.9)',
        border: `1px solid ${isUser ? 'rgba(0,200,255,.2)' : 'rgba(0,200,255,.07)'}`,
        fontSize: 13.5, lineHeight: 1.7,
        color: isUser ? 'var(--cyan)' : 'var(--text-primary)',
        fontFamily: 'var(--font-body)',
        wordBreak: 'break-word',
      }}>
        {isUser
          ? <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>
          : <div>{renderMarkdown(msg.content)}</div>
        }
        <div style={{
          fontFamily: 'var(--font-mono)', fontSize: 9,
          color: 'rgba(150,180,200,0.4)', marginTop: 6,
          textAlign: isUser ? 'right' : 'left',
        }}>{msg.timestamp}</div>
      </div>
      {isUser && (
        <div style={{
          width: 28, height: 28, borderRadius: '50%',
          background: 'rgba(0,200,255,.08)',
          border: '1px solid rgba(0,200,255,.15)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: 12, flexShrink: 0, marginLeft: 8, marginTop: 2,
        }}>👤</div>
      )}
    </div>
  );
}

function ResultBadge({ result }) {
  if (!result) return null;
  const r = result.risk || {};
  const f = result.features || {};
  const score = r.risk_score;
  const level = r.risk_level || '';
  const detected = f.tumor_detected;
  const area = f.tumor_area_cm2;
  const color = score >= 70 ? '#ff2d55' : score >= 50 ? '#ff6b35' : score >= 30 ? '#ffd60a' : '#00c896';

  return (
    <div style={{
      margin: '8px 14px',
      padding: '10px 14px',
      borderRadius: 12,
      background: 'rgba(0,0,0,0.3)',
      border: `1px solid ${color}33`,
      display: 'grid',
      gridTemplateColumns: '1fr 1fr 1fr',
      gap: 8,
    }}>
      <div style={{ textAlign: 'center' }}>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'rgba(150,180,200,0.5)', marginBottom: 3 }}>RISK SCORE</div>
        <div style={{ fontFamily: 'var(--font-display)', fontSize: 18, fontWeight: 800, color }}>{score ?? '–'}</div>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 8, color: 'rgba(150,180,200,0.4)' }}>/100</div>
      </div>
      <div style={{ textAlign: 'center', borderLeft: '1px solid rgba(0,200,255,0.08)', borderRight: '1px solid rgba(0,200,255,0.08)' }}>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'rgba(150,180,200,0.5)', marginBottom: 3 }}>MỨC ĐỘ</div>
        <div style={{ fontFamily: 'var(--font-body)', fontSize: 13, fontWeight: 700, color }}>{level || '–'}</div>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 8, color: detected ? '#ff6b35' : '#00c896' }}>
          {detected ? '● Phát hiện u' : '● Không có u'}
        </div>
      </div>
      <div style={{ textAlign: 'center' }}>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'rgba(150,180,200,0.5)', marginBottom: 3 }}>DIỆN TÍCH</div>
        <div style={{ fontFamily: 'var(--font-display)', fontSize: 18, fontWeight: 800, color: 'var(--text-primary)' }}>
          {area != null ? area : '–'}
        </div>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 8, color: 'rgba(150,180,200,0.4)' }}>cm²</div>
      </div>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────
export default function MedicalChatbot({ isOpen, onToggle, analysisResult }) {
  const hasResult = Boolean(analysisResult);
  const ts = () => new Date().toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' });

  const makeWelcome = (withResult) => ({
    role: 'assistant',
    timestamp: ts(),
    content: withResult
      ? '✅ **Kết quả phân tích đã được liên kết!**\n\nTôi đã đọc toàn bộ dữ liệu MRI từ hệ thống BrainAI.\n\nBạn có thể hỏi tôi về:\n• Mức độ nguy hiểm của khối u\n• Diện tích và vị trí u\n• Ý nghĩa từng chỉ số\n• Khuyến nghị điều trị\n• Các bước tiếp theo\n\nHãy hỏi bất kỳ điều gì!'
      : 'Xin chào! Tôi là **Trợ lý Y khoa AI** của BrainAI.\n\nTôi có thể giúp bạn:\n• Giải thích các thuật ngữ y khoa\n• Hiểu kết quả phân tích MRI\n• Tìm hiểu về khối u não\n• Giải thích cách AI hoạt động\n\n⚠️ Thông tin tham khảo, không thay thế chẩn đoán bác sĩ.',
  });

  const [messages, setMessages] = useState([makeWelcome(false)]);
  const [input, setInput]       = useState('');
  const [loading, setLoading]   = useState(false);
  const [showQuick, setShowQuick] = useState(true);
  const [minimized, setMinimized] = useState(false);
  const bottomRef = useRef(null);
  const inputRef  = useRef(null);
  const prevResult = useRef(null);
  const sendMessageRef = useRef(null);

  // Expose sendMessage globally so RiskPanel can trigger questions
  useEffect(() => {
    window.__chatbotAsk = (question) => {
      if (sendMessageRef.current) sendMessageRef.current(question);
    };
    return () => { window.__chatbotAsk = null; };
  }, []);

  // Auto-update when new result arrives
  useEffect(() => {
    if (analysisResult && analysisResult !== prevResult.current) {
      prevResult.current = analysisResult;
      setMessages([makeWelcome(true)]);
      setShowQuick(true);
      setMinimized(false);
    }
  }, [analysisResult]); // eslint-disable-line

  // Scroll to bottom
  useEffect(() => {
    if (isOpen && !minimized)
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading, isOpen, minimized]);

  // Focus input
  useEffect(() => {
    if (isOpen && !minimized)
      setTimeout(() => inputRef.current?.focus(), 300);
  }, [isOpen, minimized]);

  const sendMessage = useCallback(async (text) => {
    const q = (text || input).trim();
    if (!q || loading) return;
    setInput('');
    setShowQuick(false);
    const now = ts();
    setMessages(prev => [...prev, { role: 'user', content: q, timestamp: now }]);
    setLoading(true);
    try {
      const history = messages.slice(-8).map(m => ({ role: m.role, content: m.content }));
      const ctx = buildContext(analysisResult);
      const reply = await sendMedicalQuestion(q, history, ctx);
      setMessages(prev => [...prev, { role: 'assistant', content: reply, timestamp: ts() }]);
    } catch {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: '⚠️ Xin lỗi, gặp sự cố kết nối. Vui lòng thử lại.',
        timestamp: ts(),
      }]);
    } finally {
      setLoading(false);
    }
  }, [input, loading, messages, analysisResult]); // eslint-disable-line

  // Keep ref in sync so RiskPanel button can call it
  useEffect(() => { sendMessageRef.current = sendMessage; }, [sendMessage]);

  const handleKey = useCallback((e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  }, [sendMessage]);

  const clearChat = () => {
    setMessages([makeWelcome(hasResult)]);
    setShowQuick(true);
  };

  if (!isOpen) return null;

  const quickList = hasResult ? QUICK_WITH_RESULT : QUICK_DEFAULT;

  return (
    <>
      <style>{`
        @keyframes typing-bounce {
          0%,60%,100% { transform:translateY(0);opacity:.35; }
          30% { transform:translateY(-5px);opacity:1; }
        }
        @keyframes msg-in {
          from { opacity:0; transform:translateY(10px) scale(.97); }
          to   { opacity:1; transform:translateY(0) scale(1); }
        }
        @keyframes chatbot-in {
          from { opacity:0; transform:translateY(24px) scale(.95); }
          to   { opacity:1; transform:translateY(0) scale(1); }
        }
        .cb-input::placeholder { color:rgba(150,180,200,0.3); }
        .cb-input:focus { outline:none; border-color:rgba(0,200,255,.4)!important; }
        .cb-quick:hover { background:rgba(0,200,255,.1)!important; border-color:rgba(0,200,255,.3)!important; color:var(--cyan)!important; transform:translateY(-1px); }
        .cb-send:hover:not(:disabled) { background:rgba(0,200,255,.2)!important; border-color:rgba(0,200,255,.4)!important; }
        .cb-send:disabled { opacity:.35; cursor:not-allowed; }
        .cb-scroll::-webkit-scrollbar { width:3px; }
        .cb-scroll::-webkit-scrollbar-thumb { background:rgba(0,200,255,.15);border-radius:3px; }
        .cb-hdr-btn:hover { opacity:1!important; }
      `}</style>

      <div style={{
        position: 'fixed', bottom: 24, right: 24, zIndex: 200,
        width: minimized ? 300 : 420,
        display: 'flex', flexDirection: 'column',
        background: 'rgba(4,10,18,.98)',
        border: '1px solid rgba(0,200,255,.15)',
        borderRadius: 18,
        boxShadow: '0 24px 64px rgba(0,0,0,.7), 0 0 0 1px rgba(0,200,255,.04), inset 0 1px 0 rgba(255,255,255,.03)',
        backdropFilter: 'blur(24px)',
        animation: 'chatbot-in .4s cubic-bezier(.34,1.56,.64,1)',
        overflow: 'hidden',
        maxHeight: minimized ? 'auto' : '640px',
      }}>

        {/* ── HEADER ── */}
        <div onClick={() => setMinimized(m => !m)} style={{
          padding: '13px 16px',
          display: 'flex', alignItems: 'center', gap: 10,
          cursor: 'pointer', flexShrink: 0,
          background: 'linear-gradient(135deg,rgba(0,200,255,.05),rgba(0,255,136,.02))',
          borderBottom: minimized ? 'none' : '1px solid rgba(0,200,255,.07)',
        }}>
          {/* avatar */}
          <div style={{
            width: 36, height: 36, borderRadius: '50%', flexShrink: 0,
            background: 'linear-gradient(135deg,rgba(0,200,255,.25),rgba(0,255,136,.12))',
            border: '1px solid rgba(0,200,255,.35)',
            display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 17,
          }}>🧠</div>

          {/* title */}
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: 14, color: 'var(--text-primary)' }}>
              Trợ lý Y khoa AI
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 5, marginTop: 2 }}>
              <div style={{
                width: 6, height: 6, borderRadius: '50%',
                background: hasResult ? '#ffd60a' : 'var(--green)',
                boxShadow: `0 0 6px ${hasResult ? '#ffd60a' : 'var(--green)'}`,
              }}/>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10,
                color: hasResult ? '#ffd60a' : 'var(--green)',
                overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {hasResult ? '● Đã liên kết kết quả phân tích' : '● Sẵn sàng hỏi đáp y tế'}
              </span>
            </div>
          </div>

          {/* buttons */}
          <div style={{ display: 'flex', gap: 2 }} onClick={e => e.stopPropagation()}>
            <button className="cb-hdr-btn" onClick={clearChat} title="Làm mới" style={{
              background: 'none', border: 'none', color: 'rgba(150,180,200,0.4)',
              cursor: 'pointer', padding: '5px 7px', borderRadius: 7, fontSize: 14,
              opacity: .6, transition: 'all .2s',
            }}>↺</button>
            <button className="cb-hdr-btn" onClick={onToggle} title="Đóng" style={{
              background: 'none', border: 'none', color: 'rgba(150,180,200,0.4)',
              cursor: 'pointer', padding: '5px 7px', borderRadius: 7, fontSize: 18, lineHeight: 1,
              opacity: .6, transition: 'all .2s',
            }}>×</button>
          </div>
        </div>

        {!minimized && (
          <>
            {/* ── RESULT BADGE ── */}
            {hasResult && <ResultBadge result={analysisResult} />}

            {/* ── MESSAGES ── */}
            <div className="cb-scroll" style={{
              flex: 1, overflowY: 'auto',
              padding: '14px 14px 4px',
              minHeight: 160,
              maxHeight: hasResult ? 300 : 360,
            }}>
              {messages.map((msg, i) => <ChatMessage key={i} msg={msg} />)}

              {loading && (
                <div style={{ display: 'flex', alignItems: 'flex-start', gap: 8, marginBottom: 14 }}>
                  <div style={{
                    width: 28, height: 28, borderRadius: '50%', flexShrink: 0,
                    background: 'linear-gradient(135deg,rgba(0,200,255,.2),rgba(0,255,136,.1))',
                    border: '1px solid rgba(0,200,255,.3)',
                    display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 13,
                  }}>🧠</div>
                  <div style={{
                    background: 'rgba(8,18,28,.9)',
                    border: '1px solid rgba(0,200,255,.07)',
                    borderRadius: '4px 16px 16px 16px', padding: '2px 14px',
                  }}><TypingDots /></div>
                </div>
              )}
              <div ref={bottomRef} />
            </div>

            {/* ── QUICK QUESTIONS ── */}
            {showQuick && messages.length <= 2 && (
              <div style={{
                padding: '10px 14px 12px',
                borderTop: '1px solid rgba(0,200,255,.06)',
                display: 'flex', flexWrap: 'wrap', gap: 6,
              }}>
                <div style={{
                  width: '100%', fontFamily: 'var(--font-mono)', fontSize: 9.5,
                  color: 'rgba(150,180,200,0.45)', marginBottom: 4, letterSpacing: '0.08em',
                }}>
                  {hasResult ? '📋 GỢI Ý CÂU HỎI VỀ KẾT QUẢ NÀY' : '💡 CÂU HỎI THƯỜNG GẶP'}
                </div>
                {quickList.map(q => (
                  <button key={q} className="cb-quick" onClick={() => sendMessage(q)} style={{
                    background: 'rgba(0,200,255,.03)',
                    border: '1px solid rgba(0,200,255,.1)',
                    borderRadius: 20, padding: '5px 11px',
                    fontFamily: 'var(--font-body)', fontSize: 11.5,
                    color: 'rgba(180,210,230,0.7)',
                    cursor: 'pointer', transition: 'all .2s',
                    textAlign: 'left', lineHeight: 1.4,
                  }}>{q}</button>
                ))}
              </div>
            )}

            {/* ── INPUT ── */}
            <div style={{
              padding: '10px 14px 12px',
              borderTop: '1px solid rgba(0,200,255,.07)',
              display: 'flex', gap: 8, alignItems: 'flex-end',
            }}>
              <textarea
                ref={inputRef}
                className="cb-input"
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKey}
                disabled={loading}
                rows={1}
                placeholder={hasResult ? 'Hỏi về kết quả phân tích MRI...' : 'Hỏi về y khoa, khối u não, MRI...'}
                style={{
                  flex: 1, resize: 'none', maxHeight: 96, overflowY: 'auto',
                  background: 'rgba(8,18,28,.7)',
                  border: '1px solid rgba(0,200,255,.12)',
                  borderRadius: 12, padding: '9px 13px',
                  fontFamily: 'var(--font-body)', fontSize: 13,
                  color: 'var(--text-primary)', lineHeight: 1.55,
                  transition: 'border-color .2s',
                }}
                onInput={e => {
                  e.target.style.height = 'auto';
                  e.target.style.height = Math.min(e.target.scrollHeight, 96) + 'px';
                }}
              />
              <button
                className="cb-send"
                onClick={() => sendMessage()}
                disabled={!input.trim() || loading}
                style={{
                  width: 40, height: 40, borderRadius: 11, flexShrink: 0,
                  background: 'rgba(0,200,255,.08)',
                  border: '1px solid rgba(0,200,255,.18)',
                  color: 'var(--cyan)', cursor: 'pointer',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  transition: 'all .2s',
                }}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
                  stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="22" y1="2" x2="11" y2="13"/>
                  <polygon points="22 2 15 22 11 13 2 9 22 2"/>
                </svg>
              </button>
            </div>

            {/* ── FOOTER ── */}
            <div style={{
              padding: '5px 14px 9px',
              fontFamily: 'var(--font-mono)', fontSize: 9,
              color: 'rgba(100,140,160,0.5)', textAlign: 'center',
              borderTop: '1px solid rgba(0,200,255,.04)',
            }}>
              ⚠️ Thông tin tham khảo • Không thay thế chẩn đoán y khoa
            </div>
          </>
        )}
      </div>
    </>
  );
}