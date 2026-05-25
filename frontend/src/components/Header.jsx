import { useState, useEffect } from 'react';
import { checkHealth } from '../services/api';

export default function Header() {
  const [online, setOnline] = useState(null);

  useEffect(() => {
    checkHealth()
      .then(() => setOnline(true))
      .catch(() => setOnline(false));
  }, []);

  return (
    <header className="header">
      <div className="header-logo">
        <div className="logo-icon">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.2">
            <path d="M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10 10-4.5 10-10"/>
            <path d="M12 8v4l3 3"/>
            <circle cx="18" cy="6" r="3" fill="white" stroke="none"/>
          </svg>
        </div>
        <span className="logo-text">Brain<span>AI</span></span>
        <span className="header-badge">v1.0 · XAI Medical</span>
      </div>

      <nav style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        {['Phân tích', 'Hướng dẫn', 'Về hệ thống'].map(item => (
          <button key={item} style={{
            background: 'none',
            border: 'none',
            color: 'var(--text-secondary)',
            fontFamily: 'var(--font-body)',
            fontSize: 13,
            cursor: 'pointer',
            padding: '6px 14px',
            borderRadius: 6,
            transition: 'color 0.2s',
          }}
            onMouseEnter={e => e.target.style.color = 'var(--text-primary)'}
            onMouseLeave={e => e.target.style.color = 'var(--text-secondary)'}
          >{item}</button>
        ))}
      </nav>

      <div className="header-status">
        <div className="status-dot" style={{
          background: online === null ? '#888' : online ? 'var(--green)' : 'var(--red-hot)',
          boxShadow: `0 0 8px ${online === null ? '#888' : online ? 'var(--green)' : 'var(--red-hot)'}`,
        }} />
        <span>
          {online === null ? 'Đang kết nối...' : online ? 'API Online' : 'API Offline'}
        </span>
      </div>
    </header>
  );
}
