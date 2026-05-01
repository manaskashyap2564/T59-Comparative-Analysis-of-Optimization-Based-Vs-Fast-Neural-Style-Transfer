import React, { createContext, useContext, useState } from 'react';
import axios from 'axios';

const AuthContext = createContext();

export function AuthProvider({ children }) {
  const [user, setUser] = useState(() => {
    // Page refresh ke baad bhi login rahe
    const saved = localStorage.getItem('stylesense_user');
    return saved ? JSON.parse(saved) : null;
  });

  const login = async (username, password) => {
    const res = await axios.post('http://localhost:5000/api/login',
      { username, password });
    const userData = {
      username: res.data.username,
      role:     res.data.role,       // "user" ya "developer"
      token:    res.data.token,
    };
    setUser(userData);
    localStorage.setItem('stylesense_user', JSON.stringify(userData));
    // Axios default header set karo taaki har request mein token jaaye
    axios.defaults.headers.common['Authorization'] = 'Bearer ' + userData.token;
    return userData;
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem('stylesense_user');
    delete axios.defaults.headers.common['Authorization'];
  };

  const isDev = user?.role === 'developer';

  return (
    <AuthContext.Provider value={{ user, login, logout, isDev }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);