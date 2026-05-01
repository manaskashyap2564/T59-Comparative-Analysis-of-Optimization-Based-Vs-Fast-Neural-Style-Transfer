import React, { useState, useEffect } from 'react';
import { AppProvider } from './context/AppContext';
import { AuthProvider, useAuth } from './context/AuthContext';
import Navbar    from './components/Navbar';
import Home      from './pages/Home';
import Compare   from './pages/Compare';
import Benchmark from './pages/Benchmark';
import Recommend from './pages/Recommend';
import Login     from './pages/Login';
import { checkHealth } from './api';
import axios from 'axios';

function AppInner() {
  const { user, logout, isDev } = useAuth();
  const [page,     setPage]   = useState('home');
  const [bkStatus, setBk]     = useState('checking');

  // Token restore on mount
  useEffect(() => {
    if (user?.token)
      axios.defaults.headers.common['Authorization'] = 'Bearer ' + user.token;
  }, [user]);

  useEffect(() => {
    const ping = () => checkHealth()
      .then(() => setBk('online'))
      .catch(() => setBk('offline'));
    ping();
    const t = setInterval(ping, 10000);
    return () => clearInterval(t);
  }, []);

  // Agar login nahi hai to Login page dikhao
  if (!user) return <Login onLogin={() => setPage('home')} />;

  // Developer nahi hai aur benchmark pe jaane ki koshish kare
  if (page === 'benchmark' && !isDev) {
    setPage('home');
    return null;
  }

  const pages = { home: Home, compare: Compare, benchmark: Benchmark, recommend: Recommend };
  const Page  = pages[page] || Home;

  return (
    <div style={{ minHeight: '100vh', background: '#0f0f13', color: '#e8e8f0' }}>
      <Navbar page={page} setPage={setPage} backendStatus={bkStatus}
        isDev={isDev} user={user} onLogout={logout} />
      <Page navigate={setPage} />
    </div>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <AppProvider>
        <AppInner />
      </AppProvider>
    </AuthProvider>
  );
}