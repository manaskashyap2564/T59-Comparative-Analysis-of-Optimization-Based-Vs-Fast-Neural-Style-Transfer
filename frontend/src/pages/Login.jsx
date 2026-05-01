import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import theme from '../styles/theme';

export default function Login({ onLogin }) {
  const { login } = useAuth();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error,    setError]    = useState('');
  const [loading,  setLoading]  = useState(false);

  const handleLogin = async (e) => {
    e.preventDefault();
    setError(''); setLoading(true);
    try {
      const userData = await login(username, password);
      onLogin(userData);
    } catch {
      setError('❌ Invalid username ya password!');
    }
    setLoading(false);
  };

  return (
    <div style={{ minHeight: '100vh', background: '#0f0f13',
      display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <div style={{ width: 380, background: theme.bg.card,
        borderRadius: theme.radius.md, padding: 40,
        border: '1px solid ' + theme.colors.border, boxShadow: '0 8px 32px #0006' }}>

        {/* Logo */}
        <div style={{ textAlign: 'center', marginBottom: 32 }}>
          <div style={{ fontSize: 28, fontWeight: 800,
            background: theme.gradient.primary,
            WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
            StyleSense
          </div>
          <div style={{ color: theme.colors.textSecondary, fontSize: 14, marginTop: 4 }}>
            Neural Style Transfer Platform
          </div>
        </div>

        <form onSubmit={handleLogin}>
          {/* Username */}
          <div style={{ marginBottom: 16 }}>
            <label style={{ fontSize: 13, color: theme.colors.textSecondary,
              display: 'block', marginBottom: 6 }}>Username</label>
            <input value={username} onChange={e => setUsername(e.target.value)}
              placeholder="user / dev"
              style={{ width: '100%', padding: '10px 14px', borderRadius: theme.radius.sm,
                background: theme.bg.secondary, border: '1px solid ' + theme.colors.border,
                color: theme.colors.textPrimary, fontSize: 14, boxSizing: 'border-box' }} />
          </div>

          {/* Password */}
          <div style={{ marginBottom: 20 }}>
            <label style={{ fontSize: 13, color: theme.colors.textSecondary,
              display: 'block', marginBottom: 6 }}>Password</label>
            <input type="password" value={password} onChange={e => setPassword(e.target.value)}
              placeholder="••••••••"
              style={{ width: '100%', padding: '10px 14px', borderRadius: theme.radius.sm,
                background: theme.bg.secondary, border: '1px solid ' + theme.colors.border,
                color: theme.colors.textPrimary, fontSize: 14, boxSizing: 'border-box' }} />
          </div>

          {error && <div style={{ color: '#ff6b6b', fontSize: 13, marginBottom: 12 }}>{error}</div>}

          <button type="submit" disabled={loading}
            style={{ width: '100%', padding: '12px 0', borderRadius: theme.radius.sm,
              border: 'none', cursor: loading ? 'not-allowed' : 'pointer',
              background: theme.gradient.primary, color: '#fff',
              fontWeight: 700, fontSize: 15, opacity: loading ? 0.7 : 1 }}>
            {loading ? '⏳ Logging in...' : '🔐 Login'}
          </button>
        </form>

        {/* Demo credentials hint */}
        <div style={{ marginTop: 20, padding: 12, background: theme.bg.secondary,
          borderRadius: theme.radius.sm, fontSize: 12, color: theme.colors.textSecondary }}>
          <div style={{ fontWeight: 600, marginBottom: 6 }}>Demo Credentials:</div>
          <div>👤 User: <code>user</code> / <code>user123</code></div>
          <div>🔧 Developer: <code>dev</code> / <code>dev123</code></div>
        </div>
      </div>
    </div>
  );
}