import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 0, // Timeout yok - uzun işlemler için
});

// API fonksiyonları
export const apiService = {
  // Soru yönetimi
  getQuestions: (params = {}) => api.get('/api/questions', { params }),
  
  // İstatistikler
  getStatistics: (params = {}) => api.get('/api/statistics', { params }),
  getStatisticsSummary: () => api.get('/api/statistics/summary'),
  getTopicDistribution: () => api.get('/api/statistics/distribution'),
  
  // MCP Soru üretimi
  predictDistribution: (targetYear = 2025) => 
    api.post('/api/predict', null, { params: { target_year: targetYear } }),
  
  generateQuestions: (params = {}) => 
    api.post('/api/generate', null, { params }),
  
  generateExam: (questionCount = 10) => 
    api.post('/api/generate-exam', null, { params: { question_count: questionCount } }),
  
  // ML Model
  trainModel: (params = {}) => api.post('/api/ml/train', null, { params }),
  
  generateWithML: (params = {}) => api.post('/api/ml/generate', null, { params }),
  
  trainAndGenerate: (params = {}) => 
    api.post('/api/ml/train-and-generate', null, { params }),
  
  predictTopic: (questionText) => 
    api.post('/api/ml/predict-topic', null, { params: { question_text: questionText } }),
  
  getMLStatus: () => api.get('/api/ml/status'),
};

export default api;
