import React from 'react';
import theme from '../styles/theme';
import StatusBadge from './StatusBadge';

const NAV = [
  { key: 'home',      label: 'Home',      devOnly: false },
  { key: 'compare',   label: 'Compare',   devOnly: false },
  { key: 'benchmark', label: 'Benchmark', devOnly: true  },
  { key: 'recommend', label: 'Recommend', devOnly: false },
];

export default function Navbar({ page, setPage, backendStatus, isDev, user, onLogout }) {
  return (
    <nav style={{
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      padding: '14px 32px', background: theme.bg.secondary,
      borderBottom: '1px solid ' + theme.colors.border,
      position: 'sticky', top: 0, zIndex: 100,
    }}>
      {/* Logo */}
      <div onClick={() => setPage('home')} style={{
        fontSize: 20, fontWeight: 800, cursor: 'pointer',
        background: theme.gradient.primary,
        WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
      }}>
        StyleSense
      </div>

      {/* Nav Links — Benchmark sirf dev ko dikhega */}
      <div style={{ display: 'flex', gap: 4 }}>
        {NAV.filter(n => !n.devOnly || isDev).map(n => (
          <button key={n.key} onClick={() => setPage(n.key)} style={{
            padding: '8px 16px', borderRadius: theme.radius.sm,
            border: 'none', cursor: 'pointer', fontSize: 14,
            background: page === n.key ? theme.bg.hover : 'none',
            color: page === n.key ? theme.colors.purple : theme.colors.textSecondary,
            fontWeight: page === n.key ? 600 : 400,
            transition: 'all 0.2s',
          }}>
            {n.label}
            {n.devOnly && <span style={{ fontSize: 10, marginLeft: 4,
              background: '#f9ca24', color: '#000', borderRadius: 4,
              padding: '1px 5px', fontWeight: 700 }}>DEV</span>}
          </button>
        ))}
      </div>

      {/* Right: Status + Role Badge + User + Logout */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <StatusBadge status={backendStatus} />

        {/* Role Badge */}
        <span style={{
          fontSize: 11, fontWeight: 700, padding: '3px 10px',
          borderRadius: 20,
          background: isDev ? '#2a1a4a' : '#1a2a3a',
          color: isDev ? '#b39ddb' : '#7ec8e3',
          border: '1px solid ' + (isDev ? '#7c4dff' : '#2a6a8a'),
        }}>
          {isDev ? '🔧 Developer' : '👤 User'}
        </span>

        {/* Username */}
        <span style={{ fontSize: 13, color: theme.colors.textSecondary }}>
          {user?.username}
        </span>

        {/* Logout */}
        <button onClick={onLogout} style={{
          padding: '6px 14px', borderRadius: theme.radius.sm,
          border: '1px solid ' + theme.colors.border,
          background: 'none', color: theme.colors.textSecondary,
          cursor: 'pointer', fontSize: 13,
        }}>
          Logout
        </button>
      </div>
    </nav>
  );
}