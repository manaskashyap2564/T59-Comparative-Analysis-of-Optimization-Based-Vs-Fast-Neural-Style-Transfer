import React from 'react';
import theme from '../styles/theme';

export default function StatusBadge({ status }) {
  const map = {
    online:   { color: theme.colors.green,  label: 'Backend Online' },
    offline:  { color: theme.colors.red,    label: 'Backend Offline' },
    checking: { color: theme.colors.yellow, label: 'Checking...' },
  };
  const c = map[status] || map.checking;
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 6,
      background: theme.bg.secondary,
      border: '1px solid ' + theme.colors.border,
      borderRadius: theme.radius.pill,
      padding: '4px 12px', fontSize: 13,
    }}>
      <span style={{
        width: 8, height: 8, borderRadius: '50%',
        background: c.color,
        boxShadow: status === 'online' ? '0 0 6px ' + c.color : 'none',
        display: 'inline-block',
      }} />
      <span style={{ color: c.color }}>{c.label}</span>
    </div>
  );
}
