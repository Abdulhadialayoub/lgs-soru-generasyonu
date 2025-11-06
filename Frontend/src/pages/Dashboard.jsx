import { useState, useEffect } from 'react';
import { apiService } from '../config/api';
import { BookOpen, Sparkles, Brain, TrendingUp } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import './Dashboard.css';

function Dashboard() {
  const [stats, setStats] = useState(null);
  const [mlStatus, setMlStatus] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      const [statsRes, mlRes] = await Promise.all([
        apiService.getStatisticsSummary(),
        apiService.getMLStatus()
      ]);
      
      setStats(statsRes.data.data);
      setMlStatus(mlRes.data.data);
    } catch (error) {
      console.error('Dashboard yükleme hatası:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="loading">
        <div className="spinner"></div>
      </div>
    );
  }

  const generalStats = stats?.general_statistics || {};
  const topicSummary = stats?.topic_summary || [];
  const yearSummary = stats?.year_summary || [];

  // Grafik renkleri
  const COLORS = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444', '#06b6d4', '#ec4899'];

  return (
    <div className="dashboard">
      <div className="page-header">
        <h1 className="page-title">Dashboard</h1>
        <p className="page-subtitle">LGS İngilizce Soru Üretim Sistemi</p>
      </div>

      {/* Stat Cards */}
      <div className="grid grid-cols-4 mb-4">
        <div className="stat-card">
          <div className="stat-icon" style={{ background: '#dbeafe' }}>
            <BookOpen size={24} color="#3b82f6" />
          </div>
          <div className="stat-content">
            <p className="stat-label">Toplam Soru</p>
            <h3 className="stat-value">{generalStats.total_questions || 0}</h3>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon" style={{ background: '#f3e8ff' }}>
            <Sparkles size={24} color="#8b5cf6" />
          </div>
          <div className="stat-content">
            <p className="stat-label">Konu Çeşidi</p>
            <h3 className="stat-value">{generalStats.unique_topics || 0}</h3>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon" style={{ background: '#d1fae5' }}>
            <TrendingUp size={24} color="#10b981" />
          </div>
          <div className="stat-content">
            <p className="stat-label">Yıl Aralığı</p>
            <h3 className="stat-value">
              {generalStats.earliest_year || 0} - {generalStats.latest_year || 0}
            </h3>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon" style={{ background: mlStatus?.is_trained ? '#d1fae5' : '#fee2e2' }}>
            <Brain size={24} color={mlStatus?.is_trained ? '#10b981' : '#ef4444'} />
          </div>
          <div className="stat-content">
            <p className="stat-label">ML Model</p>
            <h3 className="stat-value">
              {mlStatus?.is_trained ? `${mlStatus.model_accuracy}%` : 'Eğitilmedi'}
            </h3>
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-2 mb-4">
        {/* Konu Dağılımı */}
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Konu Dağılımı (Top 10)</h3>
          </div>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={topicSummary.slice(0, 10)}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="topic" angle={-45} textAnchor="end" height={100} fontSize={11} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="total_questions" fill="#3b82f6" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Yıllara Göre Dağılım */}
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Yıllara Göre Soru Sayısı</h3>
          </div>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={yearSummary}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="year" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="total_questions" fill="#8b5cf6" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Konu Detayları */}
      <div className="card">
        <div className="card-header">
          <h3 className="card-title">Konu Detayları</h3>
        </div>
        <div className="table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Konu</th>
                <th>Toplam Soru</th>
                <th>Ortalama %</th>
                <th>Yıl Sayısı</th>
                <th>İlk Yıl</th>
                <th>Son Yıl</th>
              </tr>
            </thead>
            <tbody>
              {topicSummary.slice(0, 15).map((topic, index) => (
                <tr key={index}>
                  <td><strong>{topic.topic}</strong></td>
                  <td>{topic.total_questions}</td>
                  <td>{topic.avg_percentage}%</td>
                  <td>{topic.years_appeared}</td>
                  <td>{topic.first_year}</td>
                  <td>{topic.last_year}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* ML Model Durumu */}
      {mlStatus?.is_trained && (
        <div className="card mt-4">
          <div className="card-header">
            <h3 className="card-title">ML Model Durumu</h3>
          </div>
          <div className="ml-status-grid">
            <div className="ml-stat">
              <p className="ml-stat-label">Model Tipi</p>
              <p className="ml-stat-value">{mlStatus.model_type}</p>
            </div>
            <div className="ml-stat">
              <p className="ml-stat-label">Accuracy</p>
              <p className="ml-stat-value">{mlStatus.model_accuracy}%</p>
            </div>
            <div className="ml-stat">
              <p className="ml-stat-label">Eğitim Verisi</p>
              <p className="ml-stat-value">{mlStatus.training_data_size} soru</p>
            </div>
            <div className="ml-stat">
              <p className="ml-stat-label">Konu Sayısı</p>
              <p className="ml-stat-value">{mlStatus.unique_topics}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default Dashboard;
