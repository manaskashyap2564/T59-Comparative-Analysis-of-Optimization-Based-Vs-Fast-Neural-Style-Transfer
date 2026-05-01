import axios from 'axios';

const BASE = 'http://localhost:5000/api';

export const checkHealth = () => axios.get(BASE + '/health');

export const getRecommendation = (payload) =>
  axios.post(BASE + '/recommend', payload);

// Normalize: backend "fast"/"optimization" nested response ko flat karta hai
function normalizeResult(nested) {
  if (!nested) return null;
  return {
    output_url:        nested.result_url,          // /api/result/xxx.jpg
    time_seconds:      nested.runtime_ms / 1000,   // ms → seconds
    output_size_bytes: nested.output_size_bytes || null,
    style_loss:        nested.style_loss,
    content_loss:      nested.content_loss,
  };
}

export const runStylize = async (contentFile, styleFile, method, iterations) => {
  const form = new FormData();
  form.append('content_image', contentFile);
  form.append('style_image',   styleFile);
  form.append('method',        method);
  if (iterations) form.append('iterations', iterations);

  const res = await axios.post(BASE + '/stylize', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });

  // Normalize response so Compare.jsx gets flat {output_url, time_seconds}
  const d = res.data;
  if (method === 'fast') {
    return { data: normalizeResult(d.fast) };
  } else if (method === 'optimization') {
    return { data: normalizeResult(d.optimization) };
  }
  // "both" — shouldn't reach here from Compare.jsx individual calls
  return res;
};
export const listStyles   = () => axios.get(BASE + '/styles');
export const getStyleUrl  = (id) => BASE + '/styles/' + id;

export const runBenchmark = (contentFile, styleFile, resolutions, iterations) => {
  const form = new FormData();
  form.append('content_image', contentFile);
  if (styleFile) form.append('style_image', styleFile);
  form.append('resolutions', resolutions || '128,256,512');
  form.append('iterations',  iterations  || 100);
  return axios.post(BASE + '/benchmark', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
};