import { useState, useEffect, useMemo } from 'react';
import { apiService } from '../config/api';
import { BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import './Statistics.css';

function Statistics() {
  const [stats, setStats] = useState(null);
  const [distribution, setDistribution] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    loadStatistics();
  }, []);

  const loadStatistics = async () => {
    try {
      setLoading(true);
      setError('');
      
      const [statsRes, distRes] = await Promise.all([
        apiService.getStatisticsSummary(),
        apiService.getTopicDistribution()
      ]);
      
      if (statsRes.data.success) {
        setStats(statsRes.data.data);
      }
      if (distRes.data.success) {
        setDistribution(distRes.data.data);
      }
    } catch (error) {
      console.error('İstatistikler yüklenemedi:', error);
      setError('İstatistikler yüklenirken hata oluştu');
    } finally {
      setLoading(false);
    }
  };

  // Memoize edilen veriler - gereksiz re-render'ları önler
  const generalStats = useMemo(() => stats?.general_statistics || {}, [stats]);
  const topicSummary = useMemo(() => (stats?.topic_summary || []).slice(0, 20), [stats]); // İlk 20 konu
  const yearSummary = useMemo(() => stats?.year_summary || [], [stats]);
  const topicDist = useMemo(() => (distribution?.topic_distribution || []).slice(0, 10), [distribution]); // İlk 10 konu

  const COLORS = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444', '#06b6d4', '#ec4899', '#84cc16'];

  if (loading) {
    return (
      <div className="loading">
        <div className="spinner"></div>
        <p className="mt-2">İstatistikler yükleniyor...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="statistics">
        <div className="alert alert-error">
          {error}
        </div>
      </div>
    );
  }

  return (
    <div className="statistics">
      <div className="page-header">
        <h1 className="page-title">İstatistikler</h1>
        <p className="page-subtitle">Detaylı LGS soru analizleri</p>
      </div>

      {/* Genel İstatistikler */}
      <div className="card mb-4">
        <div className="card-header">
          <h3 className="card-title">Genel İstatistikler</h3>
        </div>
        <div className="general-stats-grid">
          <div className="stat-box">
            <p className="stat-label">Toplam Kayıt</p>
            <p className="stat-number">{generalStats.total_records || 0}</p>
          </div>
          <div className="stat-box">
            <p className="stat-label">Toplam Soru</p>
            <p className="stat-number">{generalStats.total_questions || 0}</p>
          </div>
          <div className="stat-box">
            <p className="stat-label">Benzersiz Konu</p>
            <p className="stat-number">{generalStats.unique_topics || 0}</p>
          </div>
          <div className="stat-box">
            <p className="stat-label">Benzersiz Yıl</p>
            <p className="stat-number">{generalStats.unique_years || 0}</p>
          </div>
          <div className="stat-box">
            <p className="stat-label">İlk Yıl</p>
            <p className="stat-number">{generalStats.earliest_year || 0}</p>
          </div>
          <div className="stat-box">
            <p className="stat-label">Son Yıl</p>
            <p className="stat-number">{generalStats.latest_year || 0}</p>
          </div>
        </div>
      </div>

      {/* Grafikler */}
      <div className="grid grid-cols-2 mb-4">
        {/* Konu Dağılımı Bar Chart */}
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Konu Dağılımı (Toplam Sorular)</h3>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={topicSummary.slice(0, 12)}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="topic" angle={-45} textAnchor="end" height={100} fontSize={10} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="total_questions" fill="#3b82f6" animationDuration={500} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Yıllara Göre Dağılım */}
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Yıllara Göre Soru Sayısı</h3>
          </div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={yearSummary}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="year" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="total_questions" fill="#8b5cf6" animationDuration={500} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Konu Yüzdeleri Pie Chart */}
      {topicDist.length > 0 && (
        <div className="grid grid-cols-2 mb-4">
          <div className="card">
            <div className="card-header">
              <h3 className="card-title">Konu Yüzde Dağılımı (2020+) - Top 8</h3>
            </div>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={topicDist.slice(0, 8)}
                  dataKey="avg_percentage"
                  nameKey="topic"
                  cx="50%"
                  cy="50%"
                  outerRadius={90}
                  label={(entry) => `${entry.topic}: ${entry.avg_percentage}%`}
                  animationDuration={500}
                >
                  {topicDist.slice(0, 8).map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>

          {/* Konu Detay Tablosu */}
          <div className="card">
            <div className="card-header">
              <h3 className="card-title">Konu Detayları (2020+) - Top 10</h3>
            </div>
            <div className="topic-detail-list">
              {topicDist.map((topic, index) => (
                <div key={index} className="topic-detail-item">
                  <div className="topic-info">
                    <span className="topic-name">{topic.topic}</span>
                    <span className="topic-years">{topic.years_appeared} yıl</span>
                  </div>
                  <div className="topic-stats">
                    <span className="topic-percentage">{topic.avg_percentage}%</span>
                    <span className="topic-count">{topic.total_questions} soru</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Detaylı Tablo */}
      {topicSummary.length > 0 && (
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Top 20 Konu - Detaylı Analiz</h3>
          </div>
          <div className="table-container">
            <table className="data-table">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Konu</th>
                  <th>Toplam Soru</th>
                  <th>Ortalama %</th>
                  <th>Yıl Sayısı</th>
                  <th>İlk Yıl</th>
                  <th>Son Yıl</th>
                </tr>
              </thead>
              <tbody>
                {topicSummary.map((topic, index) => (
                  <tr key={index}>
                    <td>{index + 1}</td>
                    <td><strong>{topic.topic}</strong></td>
                    <td>{topic.total_questions}</td>
                    <td>
                      <span className="percentage-badge">
                        {topic.avg_percentage}%
                      </span>
                    </td>
                    <td>{topic.years_appeared}</td>
                    <td>{topic.first_year}</td>
                    <td>{topic.last_year}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

export default Statistics;
