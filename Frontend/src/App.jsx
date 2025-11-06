import { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import QuestionBank from './pages/QuestionBank';
import GenerateQuestions from './pages/GenerateQuestions';
import CreateExam from './pages/CreateExam';
import MLModel from './pages/MLModel';
import Statistics from './pages/Statistics';
import { apiService } from './config/api';
import './App.css';

function App() {
  const [backendStatus, setBackendStatus] = useState('checking'); // checking, connected, error
  const [errorMessage, setErrorMessage] = useState('');

  useEffect(() => {
    checkBackendConnection();
  }, []);

  const checkBackendConnection = async () => {
    try {
      // Backend'e basit bir health check
      const response = await apiService.getStatisticsSummary();
      if (response.data) {
        setBackendStatus('connected');
      } else {
        setBackendStatus('error');
        setErrorMessage('Backend yanıt vermiyor');
      }
    } catch (error) {
      console.error('Backend bağlantı hatası:', error);
      setBackendStatus('error');
      setErrorMessage(error.message || 'Backend bağlantısı kurulamadı');
    }
  };

  // Yükleniyor ekranı
  if (backendStatus === 'checking') {
    return (
      <div className="backend-check-screen">
        <div className="backend-check-content">
          <div className="spinner-large"></div>
          <h2>Backend Bağlantısı Kontrol Ediliyor...</h2>
          <p>Lütfen bekleyin</p>
        </div>
      </div>
    );
  }

  // Hata ekranı
  if (backendStatus === 'error') {
    return (
      <div className="backend-error-screen">
        <div className="backend-error-content">
          <div className="error-icon">⚠️</div>
          <h1>Backend Bağlantısı Kurulamadı</h1>
          <p className="error-message">{errorMessage}</p>
          <div className="error-details">
            <h3>Lütfen kontrol edin:</h3>
            <ul>
              <li>Backend sunucusu çalışıyor mu? <code>python Backend/main.py</code></li>
              <li>Backend adresi doğru mu? <code>{import.meta.env.VITE_API_URL || 'http://localhost:8000'}</code></li>
              <li>Veritabanı bağlantısı var mı?</li>
            </ul>
          </div>
          <button className="retry-button" onClick={checkBackendConnection}>
            🔄 Tekrar Dene
          </button>
        </div>
      </div>
    );
  }

  // Normal uygulama
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Navigate to="/dashboard" replace />} />
          <Route path="dashboard" element={<Dashboard />} />
          <Route path="questions" element={<QuestionBank />} />
          <Route path="generate" element={<GenerateQuestions />} />
          <Route path="exam" element={<CreateExam />} />
          <Route path="ml-model" element={<MLModel />} />
          <Route path="statistics" element={<Statistics />} />
        </Route>
      </Routes>
    </Router>
  );
}

export default App;
