import React, { useState, useEffect } from 'react';
import theme from '../styles/theme';
import { runStylize, listStyles, getStyleUrl } from '../api';
import { useApp } from '../context/AppContext';
import { formatTime } from '../utils/helpers';

export default function Compare({ navigate }) {
  const { setLastCompareResult, setLastContentFile, lastStyleId, setLastStyleId } = useApp();

  const [contentFile, setContentFile] = useState(null);
  const [customStyleFile, setCustomStyleFile] = useState(null);
  const [useCustom, setUseCustom] = useState(false);
  const [presets, setPresets]     = useState([]);
  const [iterations, setIterations] = useState(300);
  const [fastResult, setFastResult] = useState(null);
  const [optResult,  setOptResult]  = useState(null);
  const [speedup,    setSpeedup]    = useState(null);
  const [loading,    setLoading]    = useState(false);
  const [error,      setError]      = useState('');
  const API_ROOT = process.env.REACT_APP_API_URL || 'http://localhost:5000';

  useEffect(() => {
    listStyles().then(r => setPresets(r.data.presets)).catch(() => {});
  }, []);

  const getStyleFile = async () => {
    if (useCustom && customStyleFile) return customStyleFile;
    const url = getStyleUrl(lastStyleId);
    const resp = await fetch(url);
    const blob = await resp.blob();
    return new File([blob], lastStyleId + '.jpg', { type: 'image/jpeg' });
  };

  const run = async (method) => {
    if (!contentFile) { setError('Content image upload karo!'); return; }
    setError(''); setLoading(true);
    setFastResult(null); setOptResult(null); setSpeedup(null);
    try {
      const styleFile = await getStyleFile();
      if (method === 'fast' || method === 'both') {
        const r = await runStylize(contentFile, styleFile, 'fast', iterations);
        setFastResult(r.data);
      }
      if (method === 'optimization' || method === 'both') {
        const r = await runStylize(contentFile, styleFile, 'optimization', iterations);
        setOptResult(r.data);
      }
      if (method === 'both') {
        const fr = (await runStylize(contentFile, styleFile, 'fast', iterations)).data;
        const or = (await runStylize(contentFile, styleFile, 'optimization', iterations)).data;
        setFastResult(fr); setOptResult(or);
        const sp = (or.time_seconds / Math.max(fr.time_seconds, 0.001)).toFixed(1);
        setSpeedup(sp);
        setLastCompareResult({ fast: fr, opt: or, speedup: sp, style: lastStyleId });
        setLastContentFile(contentFile);
      }
    } catch(e) {
      setError('Error: ' + (e?.response?.data?.error || e.message));
    }
    setLoading(false);
  };

  const previewUrl = contentFile ? URL.createObjectURL(contentFile) : null;

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: '32px 16px' }}>
      <h2 style={{ fontSize: 28, fontWeight: 700, marginBottom: 8,
        background: theme.gradient.primary, WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
        NST Comparison
      </h2>
      <p style={{ color: theme.colors.textSecondary, marginBottom: 24 }}>
        Upload images → adjust iterations → run NST → compare results side-by-side
      </p>

      {/* Upload Row */}
      <div style={{ display: 'flex', gap: 16, marginBottom: 20, flexWrap: 'wrap' }}>
        {/* Content Upload */}
        <label style={{ flex: 1, minWidth: 260, border: '2px dashed ' + theme.colors.border,
          borderRadius: theme.radius.md, padding: 20, cursor: 'pointer', textAlign: 'center',
          background: theme.bg.card }}>
          {previewUrl
            ? <img src={previewUrl} alt="content" style={{ maxHeight: 120, maxWidth: '100%', borderRadius: 8 }} />
            : <div style={{ color: theme.colors.textSecondary }}>📁 Click to upload Content Image</div>}
          {contentFile && <div style={{ fontSize: 12, color: '#7bed9f', marginTop: 6 }}>✓ {contentFile.name}</div>}
          <input type="file" accept="image/*" style={{ display: 'none' }}
            onChange={e => { setContentFile(e.target.files[0]); setLastContentFile(e.target.files[0]); }} />
        </label>

        {/* Style Selector */}
        <div style={{ flex: 1, minWidth: 260, border: '1px solid ' + theme.colors.border,
          borderRadius: theme.radius.md, padding: 20, background: theme.bg.card }}>
          <div style={{ fontSize: 13, color: theme.colors.textSecondary, marginBottom: 10 }}>🎨 Style Image</div>

          {/* Preset Dropdown */}
          <select
            value={useCustom ? 'custom' : lastStyleId}
            onChange={e => {
              if (e.target.value === 'custom') { setUseCustom(true); }
              else { setUseCustom(false); setLastStyleId(e.target.value); }
            }}
            style={{ width: '100%', padding: '8px 12px', borderRadius: theme.radius.sm,
              background: theme.bg.secondary, color: theme.colors.textPrimary,
              border: '1px solid ' + theme.colors.border, marginBottom: 12, fontSize: 14 }}>
            {presets.map(p => (
              <option key={p.id} value={p.id}>{p.name}</option>
            ))}
            <option value="custom">📤 Upload Custom Style</option>
          </select>

          {/* Preset Preview */}
          {!useCustom && (
            <img src={API_ROOT + '/api/styles/'+ lastStyleId}
              alt="style preview"
              style={{ width: '100%', maxHeight: 160, objectFit: 'contain', borderRadius: 6,
  background: '#111', padding: 4 }} />
          )}

          {/* Custom Upload */}
          {useCustom && (
            <label style={{ display: 'block', border: '2px dashed ' + theme.colors.border,
              borderRadius: 6, padding: 12, cursor: 'pointer', textAlign: 'center' }}>
              {customStyleFile
                ? <span style={{ color: '#7bed9f', fontSize: 12 }}>✓ {customStyleFile.name}</span>
                : <span style={{ color: theme.colors.textSecondary, fontSize: 13 }}>📁 Upload style image</span>}
              <input type="file" accept="image/*" style={{ display: 'none' }}
                onChange={e => setCustomStyleFile(e.target.files[0])} />
            </label>
          )}
        </div>
      </div>

      {/* Iterations Slider */}
      <div style={{ background: theme.bg.card, borderRadius: theme.radius.md,
        padding: 20, marginBottom: 20, border: '1px solid ' + theme.colors.border }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
          <span style={{ color: theme.colors.textSecondary }}>Optimization Iterations</span>
          <span style={{ color: theme.colors.purple, fontWeight: 700 }}>{iterations}</span>
        </div>
        <input type="range" min={50} max={600} step={50} value={iterations}
          onChange={e => setIterations(+e.target.value)}
          style={{ width: '100%', accentColor: theme.colors.purple }} />
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11,
          color: theme.colors.textSecondary, marginTop: 4 }}>
          <span>50 (fast, rough)</span><span>600 (slow, quality)</span>
        </div>
        <div style={{ fontSize: 12, color: theme.colors.textSecondary, marginTop: 4 }}>
          Fast NST ignores this (always ~160ms). Affects only Optimization NST.
        </div>
      </div>

      {/* Buttons */}
      {error && <div style={{ color: '#ff6b6b', marginBottom: 12, fontSize: 14 }}>{error}</div>}
      <div style={{ display: 'flex', gap: 12, marginBottom: 20 }}>
        {[['Run Fast NST','fast'],['Run Both','both'],['Run Opt NST','optimization']].map(([label, method]) => (
          <button key={method} onClick={() => run(method)} disabled={loading}
            style={{ flex: 1, padding: '12px 0', borderRadius: theme.radius.sm, border: 'none',
              cursor: loading ? 'not-allowed' : 'pointer', fontWeight: 600, fontSize: 15,
              background: method === 'both' ? theme.gradient.primary : theme.bg.hover,
              color: theme.colors.textPrimary, opacity: loading ? 0.6 : 1 }}>
            {loading ? '⏳ Running...' : label}
          </button>
        ))}
      </div>

      {/* Speedup Badge */}
      {speedup && (
        <div style={{ background: 'linear-gradient(90deg,#1a3a1a,#0f2a0f)',
          border: '1px solid #2d5a2d', borderRadius: theme.radius.md,
          padding: '14px 24px', marginBottom: 20, textAlign: 'center',
          color: '#7bed9f', fontWeight: 700, fontSize: 16 }}>
          ⚡ Fast NST is {speedup}x faster than Optimization NST!
        </div>
      )}

      {/* Results Row */}
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
        {[
          { label: '[I] Input (Content)', url: previewUrl, time: null, name: 'input' },
          { label: '[F] Fast NST Output', url: fastResult ? API_ROOT + fastResult.output_url : null, time: fastResult?.time_seconds, name: 'fast' },
          { label: '[O] Optimization NST Output', url: optResult ? API_ROOT + optResult.output_url : null, time: optResult?.time_seconds, name: 'opt' },
        ].map(({ label, url, time }) => (
          <div key={label} style={{ flex: 1, minWidth: 220, background: theme.bg.card,
            borderRadius: theme.radius.md, padding: 16, border: '1px solid ' + theme.colors.border,
            textAlign: 'center' }}>
            <div style={{ fontSize: 12, color: theme.colors.textSecondary, marginBottom: 10 }}>{label}</div>
            {url
              ? <img src={url} alt={label} style={{ width: '100%', borderRadius: 6, maxHeight: 200, objectFit: 'cover' }} />
              : <div style={{ height: 180, background: theme.bg.secondary, borderRadius: 6,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  color: theme.colors.textSecondary, fontSize: 13 }}>
                  {loading ? '⏳ Processing...' : 'No output yet'}
                </div>}
            <div style={{ fontSize: 12, color: theme.colors.textSecondary, marginTop: 8 }}>
              Time: {time ? formatTime(time) : '--'} &nbsp; Size: --
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}