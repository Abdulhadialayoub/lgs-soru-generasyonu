import { useState, useEffect } from 'react';
import { apiService } from '../config/api';
import { Search } from 'lucide-react';
import './QuestionBank.css';

function QuestionBank() {
  const [questions, setQuestions] = useState([]);
  const [topics, setTopics] = useState([]);
  const [loading, setLoading] = useState(false);
  const [loadingTopics, setLoadingTopics] = useState(true);
  const [userAnswers, setUserAnswers] = useState({});
  const [showResults, setShowResults] = useState(false);
  const [filters, setFilters] = useState({
    year: '',
    topic: '',
    limit: 50
  });

  useEffect(() => {
    loadQuestions();
    loadTopics();
  }, []);

  const loadTopics = async () => {
    try {
      setLoadingTopics(true);
      const statsResponse = await apiService.getStatisticsSummary();
      if (statsResponse.data.success && statsResponse.data.data.topic_summary) {
        const topicList = statsResponse.data.data.topic_summary || [];
        const popularTopics = topicList.slice(0, 30).map(t => t.topic);
        setTopics(popularTopics);
      } else {
        const questionsResponse = await apiService.getQuestions({ limit: 200 });
        if (questionsResponse.data.success) {
          const questions = questionsResponse.data.data || [];
          const uniqueTopics = [...new Set(questions.map(q => q.topic))].filter(Boolean);
          setTopics(uniqueTopics.sort());
        }
      }
    } catch (err) {
      console.error('Konular yüklenemedi:', err);
      setTopics([
        'Teen Life',
        'Friendship',
        'The Internet',
        'Adventures',
        'Tourism',
        'On The Phone',
        'Science',
        'In The Kitchen',
        'Sports',
        'Movies'
      ]);
    } finally {
      setLoadingTopics(false);
    }
  };

  const loadQuestions = async () => {
    try {
      setLoading(true);
      
      // Parametreleri hazırla - limit her zaman gönder
      const cleanFilters = { limit: filters.limit || 50 };
      if (filters.year) cleanFilters.year = parseInt(filters.year);
      if (filters.topic) cleanFilters.topic = filters.topic;
      
      console.log('API çağrısı yapılıyor:', cleanFilters);
      const response = await apiService.getQuestions(cleanFilters);
      console.log('API yanıtı:', response.data);
      
      if (response.data.success) {
        setQuestions(response.data.data || []);
      }
    } catch (error) {
      console.error('Sorular yüklenemedi:', error);
      alert('Sorular yüklenirken hata oluştu. Backend çalışıyor mu?');
    } finally {
      setLoading(false);
    }
  };

  const handleFilterChange = (e) => {
    const { name, value } = e.target;
    setFilters(prev => ({ ...prev, [name]: value }));
  };

  const handleSearch = () => {
    loadQuestions();
  };

  const handleReset = () => {
    setFilters({ year: '', topic: '', limit: 50 });
    setTimeout(loadQuestions, 100);
  };

  const handleAnswerSelect = (questionId, answer) => {
    if (!showResults) {
      setUserAnswers(prev => ({ ...prev, [questionId]: answer }));
    }
  };

  const handleSubmitExam = () => {
    setShowResults(true);
  };

  const calculateScore = () => {
    let correct = 0;
    questions.forEach(q => {
      if (userAnswers[q.id] === q.correct_option) {
        correct++;
      }
    });
    return { 
      correct, 
      total: questions.length, 
      percentage: questions.length > 0 ? ((correct / questions.length) * 100).toFixed(1) : 0 
    };
  };

  return (
    <div className="question-bank">
      <div className="page-header">
        <h1 className="page-title">Soru Bankası</h1>
        <p className="page-subtitle">Geçmiş LGS İngilizce sorularını inceleyin</p>
      </div>

      {/* Filters */}
      <div className="card mb-4">
        <div className="filters-grid">
          <div className="form-group">
            <label className="form-label">Yıl</label>
            <input
              type="number"
              name="year"
              className="form-input"
              placeholder="Örn: 2024"
              value={filters.year}
              onChange={handleFilterChange}
            />
          </div>

          <div className="form-group">
            <label className="form-label">Konu</label>
            <select
              name="topic"
              className="form-select"
              value={filters.topic}
              onChange={handleFilterChange}
              disabled={loadingTopics}
            >
              <option value="">Tüm Konular</option>
              {topics.map((topic, index) => (
                <option key={index} value={topic}>
                  {topic}
                </option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Limit</label>
            <select
              name="limit"
              className="form-select"
              value={filters.limit}
              onChange={handleFilterChange}
            >
              <option value="25">25 soru</option>
              <option value="50">50 soru</option>
              <option value="100">100 soru</option>
              <option value="200">200 soru</option>
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">&nbsp;</label>
            <div className="flex gap-2">
              <button className="btn btn-primary" onClick={handleSearch}>
                <Search size={18} />
                Ara
              </button>
              <button className="btn btn-outline" onClick={handleReset}>
                Sıfırla
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Results */}
      <div className="card">
        <div className="card-header flex justify-between items-center">
          <h3 className="card-title">
            {questions.length} Soru Bulundu
          </h3>
          <div className="flex gap-2">
            {!showResults && Object.keys(userAnswers).length > 0 && (
              <button className="btn btn-primary" onClick={handleSubmitExam}>
                Cevapları Kontrol Et ({Object.keys(userAnswers).length}/{questions.length})
              </button>
            )}
            {showResults && (
              <button className="btn btn-outline" onClick={() => {
                setUserAnswers({});
                setShowResults(false);
              }}>
                Sıfırla
              </button>
            )}
          </div>
        </div>

        {showResults && (
          <div className="exam-results">
            <h4>Sınav Sonucu</h4>
            <div className="score-display">
              <div className="score-item">
                <span className="score-label">Doğru:</span>
                <span className="score-value correct">{calculateScore().correct}</span>
              </div>
              <div className="score-item">
                <span className="score-label">Yanlış:</span>
                <span className="score-value wrong">{calculateScore().total - calculateScore().correct}</span>
              </div>
              <div className="score-item">
                <span className="score-label">Başarı:</span>
                <span className="score-value percentage">{calculateScore().percentage}%</span>
              </div>
            </div>
          </div>
        )}

        {loading ? (
          <div className="loading">
            <div className="spinner"></div>
          </div>
        ) : (
          <div className="questions-list">
            {questions.map((q, index) => (
              <div key={q.id} className="question-item">
                <div className="question-header">
                  <span className="question-number">#{index + 1}</span>
                  <span className="badge badge-primary">{q.topic}</span>
                  <span className="badge badge-success">{q.year}</span>
                </div>
                
                <div className="question-text">
                  <strong>Soru:</strong> {q.question_text}
                </div>

                <div className="options-grid">
                  {['A', 'B', 'C', 'D'].map(option => {
                    const optionText = q[`option_${option.toLowerCase()}`];
                    const isSelected = userAnswers[q.id] === option;
                    const isCorrect = q.correct_option === option;
                    
                    let optionClass = 'option';
                    if (isSelected && !showResults) {
                      optionClass += ' selected';
                    }
                    if (showResults) {
                      if (isCorrect) {
                        optionClass += ' correct';
                      } else if (isSelected && !isCorrect) {
                        optionClass += ' wrong';
                      }
                    }
                    
                    return (
                      <div 
                        key={option}
                        className={optionClass}
                        onClick={() => !showResults && handleAnswerSelect(q.id, option)}
                        style={{ cursor: !showResults ? 'pointer' : 'default' }}
                      >
                        <strong>{option})</strong> {optionText}
                      </div>
                    );
                  })}
                </div>

                {showResults && (
                  <div className="question-footer">
                    <span className="correct-answer">
                      Doğru Cevap: <strong>{q.correct_option}</strong>
                    </span>
                    {userAnswers[q.id] && (
                      <span className={`user-answer ${userAnswers[q.id] === q.correct_option ? 'correct' : 'wrong'}`}>
                        Sizin Cevabınız: <strong>{userAnswers[q.id]}</strong>
                        {userAnswers[q.id] === q.correct_option ? ' (Doğru)' : ' (Yanlış)'}
                      </span>
                    )}
                  </div>
                )}
              </div>
            ))}

            {questions.length === 0 && !loading && (
              <div className="empty-state">
                <p>Soru bulunamadı. Filtreleri değiştirip tekrar deneyin.</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default QuestionBank;
