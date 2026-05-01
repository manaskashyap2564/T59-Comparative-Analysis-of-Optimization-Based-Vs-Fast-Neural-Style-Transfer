export const formatTime = (seconds) => {
  if (!seconds) return '--';
  if (seconds < 1) return (seconds * 1000).toFixed(0) + 'ms';
  return seconds.toFixed(2) + 's';
};

export const formatSize = (bytes) => {
  if (!bytes) return '--';
  return (bytes / 1024).toFixed(1) + ' KB';
};

export const calcSpeedup = (optTime, fastTime) => {
  if (!optTime || !fastTime) return '--';
  return (optTime / fastTime).toFixed(1) + 'x';
};