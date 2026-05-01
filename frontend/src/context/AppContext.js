import React, { createContext, useContext, useState, useEffect } from 'react';

const AppContext = createContext();

export function AppProvider({ children }) {
  const [lastCompareResult, setLastCompareResult] = useState(() => {
    try { return JSON.parse(localStorage.getItem('lastCompareResult')) || null; }
    catch { return null; }
  });
  const [lastContentFile, setLastContentFile] = useState(null);
  const [lastStyleId, setLastStyleId] = useState('vangogh');

  useEffect(() => {
    if (lastCompareResult)
      localStorage.setItem('lastCompareResult', JSON.stringify(lastCompareResult));
  }, [lastCompareResult]);

  return (
    <AppContext.Provider value={{
      lastCompareResult, setLastCompareResult,
      lastContentFile,   setLastContentFile,
      lastStyleId,       setLastStyleId,
    }}>
      {children}
    </AppContext.Provider>
  );
}

export const useApp = () => useContext(AppContext);