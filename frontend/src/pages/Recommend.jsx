import React, { useState } from 'react';
import { getRecommendation } from '../api';
import theme from '../styles/theme';

const QUESTIONS = [
  {
    key: 'speed_priority',
    question: 'How important is speed to you?',
    options: [
      { value: 'realtime', label: 'Real-time / instant (< 1 second)' },
      { value: 'few_seconds', label: 'A few seconds is fine' },
      { value: 'quality_first', label: 'I want best quality, time does not matter' },
    ],
  },
  {
    key: 'use_case',
    question: 'What is your use case?',
    options: [
      { value: 'bulk', label: 'Processing many images (bulk / video frames)' },
      { value: 'single', label: 'Single high-quality image for a project' },
      { value: 'explore', label: 'Just exploring / experimenting' },
    ],
  },
  {
    key: 'style_flexibility',
    question: 'Do you need flexible style input?',
    options: [
      { value: 'fixed', label: 'One fixed style is fine (e.g. Van Gogh always)' },
      { value: 'any', label: 'I want to use any style image I choose' },
    ],
  },
];

function localRecommend(answers) {
  const { speed_priority, use_case, style_flexibility } = answers;

  if (
    speed_priority === 'realtime' ||
    use_case === 'bulk' ||
    style_flexibility === 'fixed'
  ) {
    return {
      method: 'Fast NST',
      reason: 'Fast NST is ideal for real-time, bulk processing, or when a fixed style is acceptable. It runs in ~160ms per image after one-time training.',
      color: theme.colors.purple,
    };
  }

  if (
    speed_priority === 'quality_first' ||
    use_case === 'single' ||
    style_flexibility === 'any'
  ) {
    return {
      method: 'Optimization NST',
      reason: 'Optimization NST gives higher quality output and works with any style image without retraining. Best for single high-quality renders.',
      color: theme.colors.blue,
    };
  }

  return {
    method: 'Either works!',
    reason: 'Both methods are suitable for your use case. Try Fast NST first for speed, then Optimization NST if you need better quality.',
    color: theme.colors.green,
  };
}

export default function Recommend({ backendStatus }) {
  const [answers, setAnswers] = useState({});
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const allAnswered = QUESTIONS.every(q => answers[q.key]);

  const handleSubmit = async () => {
    setLoading(true);
    try {
      const res = await getRecommendation(answers);
      setResult(res.data);
    } catch {
      setResult(localRecommend(answers));
    }
    setLoading(false);
  };

  const handleReset = () => {
    setAnswers({});
    setResult(null);
  };

  return (
    <div style={{ maxWidth: 700, margin: '0 auto' }}>
      <h2 style={{ fontSize: 28, fontWeight: 700, marginBottom: 8 }}>Get Recommendation</h2>
      <p style={{ color: theme.colors.textSecondary, marginBottom: 32 }}>
        Answer a few questions and we will suggest the best NST method for your use case.
      </p>

      {!result ? (
        <>
          {QUESTIONS.map((q, qi) => (
            <div key={q.key} style={{
              background: theme.bg.secondary,
              border: `1px solid ${theme.colors.border}`,
              borderRadius: theme.radius.md, padding: 24, marginBottom: 16,
            }}>
              <p style={{ fontSize: 15, fontWeight: 600, marginBottom: 16,
                          color: theme.colors.textPrimary }}>
                {qi + 1}. {q.question}
              </p>
              {q.options.map(opt => (
                <label key={opt.value} style={{
                  display: 'flex', alignItems: 'center', gap: 12,
                  padding: '10px 16px', marginBottom: 8,
                  background: answers[q.key] === opt.value ? theme.bg.hover : theme.bg.card,
                  border: `1px solid ${answers[q.key] === opt.value ? theme.colors.purple : theme.colors.border}`,
                  borderRadius: theme.radius.sm, cursor: 'pointer',
                }}>
                  <input
                    type="radio"
                    name={q.key}
                    value={opt.value}
                    checked={answers[q.key] === opt.value}
                    onChange={() => setAnswers(prev => ({ ...prev, [q.key]: opt.value }))}
                    style={{ accentColor: theme.colors.purple }}
                  />
                  <span style={{ fontSize: 14, color: theme.colors.textPrimary }}>
                    {opt.label}
                  </span>
                </label>
              ))}
            </div>
          ))}

          <button
            onClick={handleSubmit}
            disabled={!allAnswered || loading}
            style={{
              width: '100%', padding: '14px', fontSize: 16, fontWeight: 700,
              background: allAnswered ? theme.gradient.primary : theme.bg.hover,
              border: 'none', borderRadius: theme.radius.sm, color: '#fff',
              cursor: allAnswered ? 'pointer' : 'not-allowed',
            }}>
            {loading ? 'Analyzing...' : 'Get My Recommendation'}
          </button>
        </>
      ) : (
        <div style={{
          background: theme.bg.secondary,
          border: `2px solid ${result.color || theme.colors.purple}`,
          borderRadius: theme.radius.lg, padding: 40, textAlign: 'center',
        }}>
          <div style={{ fontSize: 48, marginBottom: 16 }}>
            {result.method === 'Fast NST' ? 'F' : result.method === 'Optimization NST' ? 'O' : '*'}
          </div>
          <h3 style={{
            fontSize: 28, fontWeight: 800, marginBottom: 16,
            color: result.color || theme.colors.purple,
          }}>
            {result.method}
          </h3>
          <p style={{
            fontSize: 15, color: theme.colors.textSecondary,
            lineHeight: 1.7, marginBottom: 32,
          }}>
            {result.reason}
          </p>
          <button onClick={handleReset} style={{
            padding: '12px 32px', fontSize: 14, fontWeight: 600,
            background: theme.bg.card,
            border: `1px solid ${theme.colors.border}`,
            borderRadius: theme.radius.sm,
            color: theme.colors.textPrimary, cursor: 'pointer',
          }}>
            Start Over
          </button>
        </div>
      )}
    </div>
  );
}