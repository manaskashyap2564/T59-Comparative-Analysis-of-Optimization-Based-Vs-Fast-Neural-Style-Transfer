import React from 'react';
import theme from '../styles/theme';

const FEATURES = [
  { icon: '⚡', title: '522x Speedup',   desc: 'Fast NST vs Optimization NST' },
  { icon: '🖼️', title: 'Side-by-Side',   desc: 'Compare output quality visually' },
  { icon: '🤖', title: 'Smart Recommend', desc: 'Best method for your use-case' },
];

export default function Home({ navigate }) {
  return (
    <div style={{ maxWidth: 800, margin: '0 auto', textAlign: 'center', paddingTop: 60 }}>

      <div style={{ fontSize: 64, marginBottom: 16 }}>🎨</div>

      <h1 style={{
        fontSize: 42, fontWeight: 800, marginBottom: 12,
        background: theme.gradient.primary,
        WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
      }}>
        StyleSense
      </h1>

      <p style={{ color: theme.colors.textSecondary, fontSize: 18, marginBottom: 8 }}>
        Optimization-Based vs Fast Neural Style Transfer
      </p>
      <p style={{ color: theme.colors.textMuted, fontSize: 14, marginBottom: 48 }}>
        Team T59 · GLA University · B.Tech CSE AI/ML
      </p>

      {/* Feature Cards */}
      <div style={{
        display: 'grid', gridTemplateColumns: 'repeat(3,1fr)',
        gap: 16, marginBottom: 48,
      }}>
        {FEATURES.map(f => (
          <div key={f.title} style={{
            background: theme.bg.secondary,
            border: '1px solid ' + theme.colors.border,
            borderRadius: theme.radius.md, padding: 24,
          }}>
            <div style={{ fontSize: 32, marginBottom: 8 }}>{f.icon}</div>
            <h3 style={{ fontSize: 16, marginBottom: 6, color: theme.colors.textPrimary }}>
              {f.title}
            </h3>
            <p style={{ fontSize: 13, color: theme.colors.textSecondary }}>{f.desc}</p>
          </div>
        ))}
      </div>

      {/* CTA Buttons */}
      <button onClick={() => navigate('compare')} style={{
        padding: '14px 40px', fontSize: 16, fontWeight: 700,
        background: theme.gradient.primary,
        border: 'none', borderRadius: theme.radius.sm,
        color: '#fff', cursor: 'pointer', marginRight: 12,
      }}>
        🚀 Start Comparing
      </button>
      <button onClick={() => navigate('recommend')} style={{
        padding: '14px 40px', fontSize: 16, fontWeight: 600,
        background: theme.bg.card,
        border: '1px solid ' + theme.colors.border,
        borderRadius: theme.radius.sm,
        color: theme.colors.textPrimary, cursor: 'pointer',
      }}>
        🤖 Get Recommendation
      </button>
    </div>
  );
}
