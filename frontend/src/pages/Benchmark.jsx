import React, { useState } from 'react';
import theme from '../styles/theme';
import { runBenchmark } from '../api';
import { useApp } from '../context/AppContext';
import { formatTime } from '../utils/helpers';

export default function Benchmark({ navigate }) {
  const { lastCompareResult, lastContentFile } = useApp();
  const [results,  setResults]  = useState(null);
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState('');
  const [contentFile, setContentFile] = useState(null);

  const effectiveContent = contentFile || lastContentFile;

  const run = async () => {
    if (!effectiveContent) { setError('Content image chahiye — Compare page se aao ya upload karo!'); return; }
    setError(''); setLoading(true); setResults(null);
    try {
      const r = await runBenchmark(effectiveContent, null, '128,256,512', 100);
      setResults(r.data.benchmark);
    } catch(e) {
      setError('Error: ' + (e?.response?.data?.error || e.message));
    }
    setLoading(false);
  };

  return (
    <div style={{ maxWidth: 900, margin: '0 auto', padding: '32px 16px' }}>
      <h2 style={{ fontSize: 28, fontWeight: 700, marginBottom: 8,
        background: theme.gradient.primary, WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
        Benchmark
      </h2>
      <p style={{ color: theme.colors.textSecondary, marginBottom: 24 }}>
        Compare Fast vs Optimization NST at multiple resolutions
      </p>

      {/* Auto-loaded session notice */}
      {lastContentFile && !contentFile && (
        <div style={{ background: '#1a2a3a', border: '1px solid #2a4a6a',
          borderRadius: theme.radius.sm, padding: '10px 16px', marginBottom: 16,
          fontSize: 13, color: '#7ec8e3' }}>
          ✅ Compare page ki image auto-load ho gayi — seedha Run karo!
        </div>
      )}

      {/* Upload (optional override) */}
      <label style={{ display: 'block', border: '2px dashed ' + theme.colors.border,
        borderRadius: theme.radius.md, padding: 20, cursor: 'pointer', textAlign: 'center',
        background: theme.bg.card, marginBottom: 20 }}>
        {contentFile
          ? <span style={{ color: '#7bed9f' }}>✓ {contentFile.name}</span>
          : lastContentFile
            ? <span style={{ color: '#7ec8e3' }}>📎 Using: {lastContentFile.name} (Compare se) — ya naya upload karo</span>
            : <span style={{ color: theme.colors.textSecondary }}>📁 Upload Content Image</span>}
        <input type="file" accept="image/*" style={{ display: 'none' }}
          onChange={e => setContentFile(e.target.files[0])} />
      </label>

      {error && <div style={{ color: '#ff6b6b', marginBottom: 12 }}>{error}</div>}

      <button onClick={run} disabled={loading}
        style={{ width: '100%', padding: '14px 0', borderRadius: theme.radius.sm,
          border: 'none', cursor: loading ? 'not-allowed' : 'pointer', fontWeight: 700,
          fontSize: 16, background: theme.gradient.primary, color: '#fff', marginBottom: 24,
          opacity: loading ? 0.6 : 1 }}>
        {loading ? '⏳ Benchmarking all resolutions...' : '🚀 Run Benchmark'}
      </button>

      {/* Results Table */}
      {results && (
        <div style={{ background: theme.bg.card, borderRadius: theme.radius.md,
          border: '1px solid ' + theme.colors.border, overflow: 'hidden' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 14 }}>
            <thead>
              <tr style={{ background: theme.bg.secondary }}>
                {['Resolution','Fast NST','Opt NST','Speedup','Fast Loss','Opt Loss'].map(h => (
                  <th key={h} style={{ padding: '12px 16px', color: theme.colors.textSecondary,
                    fontWeight: 600, textAlign: 'center' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {results.map((r, i) => (
                <tr key={i} style={{ borderTop: '1px solid ' + theme.colors.border }}>
                  <td style={{ padding: '12px 16px', textAlign: 'center', fontWeight: 600,
                    color: theme.colors.purple }}>{r.resolution}×{r.resolution}</td>
                  <td style={{ padding: '12px 16px', textAlign: 'center', color: '#7bed9f' }}>
                    {formatTime(r.fast_runtime_ms / 1000)}</td>
                  <td style={{ padding: '12px 16px', textAlign: 'center', color: '#ffa07a' }}>
                    {formatTime(r.opt_runtime_ms  / 1000)}</td>
                  <td style={{ padding: '12px 16px', textAlign: 'center', fontWeight: 700,
                    color: '#f9ca24' }}>⚡ {r.speedup}x</td>
                  <td style={{ padding: '12px 16px', textAlign: 'center', color: theme.colors.textSecondary }}>
                    {r.fast_style_loss?.toFixed(2) ?? '--'}</td>
                  <td style={{ padding: '12px 16px', textAlign: 'center', color: theme.colors.textSecondary }}>
                    {r.opt_style_loss?.toFixed(2) ?? '--'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Compare page se results bhi dikhao */}
      {!results && lastCompareResult && (
        <div style={{ background: theme.bg.card, borderRadius: theme.radius.md,
          padding: 20, border: '1px solid ' + theme.colors.border }}>
          <div style={{ fontSize: 14, color: theme.colors.textSecondary, marginBottom: 12 }}>
            📊 Last Compare Session Result:
          </div>
          <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap' }}>
            {[
              { label: 'Fast NST', time: lastCompareResult.fast?.time_seconds, color: '#7bed9f' },
              { label: 'Opt NST',  time: lastCompareResult.opt?.time_seconds,  color: '#ffa07a' },
              { label: 'Speedup',  value: lastCompareResult.speedup + 'x',     color: '#f9ca24' },
            ].map(({ label, time, value, color }) => (
              <div key={label} style={{ flex: 1, minWidth: 140, background: theme.bg.secondary,
                borderRadius: theme.radius.sm, padding: '12px 16px', textAlign: 'center' }}>
                <div style={{ fontSize: 12, color: theme.colors.textSecondary }}>{label}</div>
                <div style={{ fontSize: 22, fontWeight: 700, color, marginTop: 4 }}>
                  {value || formatTime(time)}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}