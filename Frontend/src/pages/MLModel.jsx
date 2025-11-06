import { useState, useEffect } from 'react';
import { apiService } from '../config/api';
import { Brain, Loader, Zap, TrendingUp, FileDown } from 'lucide-react';
import jsPDF from 'jspdf';
import './MLModel.css';

function MLModel() {
  const [mlStatus, setMlStatus] = useState(null);
  const [topics, setTopics] = useState([]);
  const [loadingTopics, setLoadingTopics] = useState(true);
  const [trainParams, setTrainParams] = useState({
    topic: '',
    limit: 200,
    model_type: 'random_forest'
  });
  const [generateParams, setGenerateParams] = useState({
    topic: '',
    count: 5,
    difficulty: 'orta'
  });
  const [questions, setQuestions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [training, setTraining] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [examMode, setExamMode] = useState(true); // Varsayılan olarak açık
  const [userAnswers, setUserAnswers] = useState({});
  const [showResults, setShowResults] = useState(false);

  useEffect(() => {
    loadMLStatus();
    loadTopics();
  }, []);

  const loadTopics = async () => {
    try {
      setLoadingTopics(true);
      const statsResponse = await apiService.getStatisticsSummary();
      if (statsResponse.data.success && statsResponse.data.data.topic_summary) {
        const topicList = statsResponse.data.data.topic_summary || [];
        const popularTopics = topicList.slice(0, 20).map(t => t.topic);
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

  const loadMLStatus = async () => {
    try {
      const response = await apiService.getMLStatus();
      setMlStatus(response.data.data);
    } catch (err) {
      console.error('ML status yüklenemedi:', err);
    }
  };

  const handleTrain = async () => {
    try {
      setTraining(true);
      setError('');
      setSuccess('');
      
      const response = await apiService.trainModel(trainParams);
      
      if (response.data.success) {
        setSuccess('Model başarıyla eğitildi!');
        await loadMLStatus();
      } else {
        setError(response.data.message);
      }
    } catch (err) {
      setError('Model eğitimi hatası: ' + err.message);
    } finally {
      setTraining(false);
    }
  };

  const handleGenerate = async () => {
    try {
      setLoading(true);
      setError('');
      setSuccess('');
      
      const response = await apiService.generateWithML(generateParams);
      
      if (response.data.success) {
        setQuestions(response.data.questions);
        setSuccess(response.data.message);
      } else {
        setError(response.data.message);
      }
    } catch (err) {
      setError('Soru üretimi hatası: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleTrainAndGenerate = async () => {
    try {
      setTraining(true);
      setLoading(true);
      setError('');
      setSuccess('');
      
      const response = await apiService.trainAndGenerate({
        ...trainParams,
        question_count: generateParams.count,
        difficulty: generateParams.difficulty
      });
      
      if (response.data.success) {
        setQuestions(response.data.questions);
        setSuccess(response.data.message);
        await loadMLStatus();
      } else {
        setError(response.data.message);
      }
    } catch (err) {
      setError('İşlem hatası: ' + err.message);
    } finally {
      setTraining(false);
      setLoading(false);
    }
  };

  const handleDownloadPDF = () => {
    if (questions.length === 0) return;

    const doc = new jsPDF();
    doc.setLanguage("tr");
    
    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();
    const margin = 20;
    const maxWidth = pageWidth - 2 * margin;
    let yPosition = margin;

    // Türkçe karakter temizleme fonksiyonu
    const cleanText = (text) => {
      if (!text) return '';
      return text
        .replace(/İ/g, 'I')
        .replace(/ı/g, 'i')
        .replace(/Ş/g, 'S')
        .replace(/ş/g, 's')
        .replace(/Ğ/g, 'G')
        .replace(/ğ/g, 'g')
        .replace(/Ü/g, 'U')
        .replace(/ü/g, 'u')
        .replace(/Ö/g, 'O')
        .replace(/ö/g, 'o')
        .replace(/Ç/g, 'C')
        .replace(/ç/g, 'c');
    };

    // Başlık
    doc.setFontSize(18);
    doc.setFont('helvetica', 'bold');
    doc.text('ML MODEL ILE URETILEN SORULAR', pageWidth / 2, yPosition, { align: 'center' });
    yPosition += 15;

    // Model Bilgileri
    doc.setFontSize(10);
    doc.setFont('helvetica', 'normal');
    if (mlStatus) {
      doc.text(`Model: ${mlStatus.model_type}`, margin, yPosition);
      yPosition += 7;
      doc.text(`Accuracy: ${mlStatus.model_accuracy}%`, margin, yPosition);
      yPosition += 7;
    }
    doc.text(`Toplam Soru: ${questions.length}`, margin, yPosition);
    yPosition += 15;

    // Sorular
    questions.forEach((q, index) => {
      if (yPosition > pageHeight - 60) {
        doc.addPage();
        yPosition = margin;
      }

      doc.setFontSize(12);
      doc.setFont('helvetica', 'bold');
      doc.text(`Soru ${index + 1}`, margin, yPosition);
      if (q.ml_predicted_topic) {
        doc.setFontSize(9);
        doc.setFont('helvetica', 'italic');
        doc.text(`(ML: ${cleanText(q.ml_predicted_topic)} - ${q.ml_confidence?.toFixed(1)}%)`, margin + 20, yPosition);
      }
      yPosition += 8;

      doc.setFontSize(10);
      doc.setFont('helvetica', 'normal');
      const questionLines = doc.splitTextToSize(cleanText(q.question_text), maxWidth);
      doc.text(questionLines, margin, yPosition);
      yPosition += questionLines.length * 6 + 5;

      const options = [
        { label: 'A', text: cleanText(q.option_a) },
        { label: 'B', text: cleanText(q.option_b) },
        { label: 'C', text: cleanText(q.option_c) },
        { label: 'D', text: cleanText(q.option_d) }
      ];

      options.forEach(opt => {
        if (yPosition > pageHeight - 20) {
          doc.addPage();
          yPosition = margin;
        }
        const optionLines = doc.splitTextToSize(`${opt.label}) ${opt.text}`, maxWidth - 5);
        doc.text(optionLines, margin + 3, yPosition);
        yPosition += optionLines.length * 6;
      });

      yPosition += 10;
    });

    // Cevap Anahtarı - Yeni sayfa
    doc.addPage();
    yPosition = margin;

    doc.setFontSize(16);
    doc.setFont('helvetica', 'bold');
    doc.text('CEVAP ANAHTARI', pageWidth / 2, yPosition, { align: 'center' });
    yPosition += 15;

    doc.setFontSize(10);
    doc.setFont('helvetica', 'normal');

    // Cevapları 4 sütunda göster
    const answersPerColumn = Math.ceil(questions.length / 4);
    const columnWidth = (pageWidth - 2 * margin) / 4;
    
    questions.forEach((q, index) => {
      const column = Math.floor(index / answersPerColumn);
      const row = index % answersPerColumn;
      const xPos = margin + (column * columnWidth);
      const yPos = yPosition + (row * 7);

      if (yPos > pageHeight - 30) return;

      doc.text(`${index + 1}. ${q.correct_option}`, xPos, yPos);
    });

    doc.save(`ML_Sorular_${new Date().toISOString().split('T')[0]}.pdf`);
  };

  return (
    <div className="ml-model">
      <div className="page-header">
        <h1 className="page-title">ML Model</h1>
        <p className="page-subtitle">Machine Learning ile soru üretimi</p>
      </div>

      {/* ML Status */}
      <div className="card mb-4">
        <div className="card-header">
          <h3 className="card-title">Model Durumu</h3>
        </div>
        
        {mlStatus ? (
          <div className="ml-status-grid">
            <div className="status-item">
              <div className="status-icon" style={{ background: mlStatus.is_trained ? '#d1fae5' : '#fee2e2' }}>
                <Brain size={24} color={mlStatus.is_trained ? '#10b981' : '#ef4444'} />
              </div>
              <div>
                <p className="status-label">Durum</p>
                <p className="status-value">
                  {mlStatus.is_trained ? 'Eğitildi' : 'Eğitilmedi'}
                </p>
              </div>
            </div>

            {mlStatus.is_trained && (
              <>
                <div className="status-item">
                  <div className="status-icon" style={{ background: '#dbeafe' }}>
                    <TrendingUp size={24} color="#3b82f6" />
                  </div>
                  <div>
                    <p className="status-label">Accuracy</p>
                    <p className="status-value">{mlStatus.model_accuracy}%</p>
                  </div>
                </div>

                <div className="status-item">
                  <div className="status-icon" style={{ background: '#f3e8ff' }}>
                    <Zap size={24} color="#8b5cf6" />
                  </div>
                  <div>
                    <p className="status-label">Model Tipi</p>
                    <p className="status-value">{mlStatus.model_type}</p>
                  </div>
                </div>

                <div className="status-item">
                  <div className="status-icon" style={{ background: '#fef3c7' }}>
                    <Brain size={24} color="#f59e0b" />
                  </div>
                  <div>
                    <p className="status-label">Eğitim Verisi</p>
                    <p className="status-value">{mlStatus.training_data_size} soru</p>
                  </div>
                </div>
              </>
            )}
          </div>
        ) : (
          <div className="loading">
            <div className="spinner"></div>
          </div>
        )}
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* Train Model */}
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Model Eğitimi</h3>
          </div>

          <div className="form-group">
            <label className="form-label">Konu (Opsiyonel)</label>
            <select
              className="form-select"
              value={trainParams.topic}
              onChange={(e) => setTrainParams({ ...trainParams, topic: e.target.value })}
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
            <label className="form-label">Eğitim Verisi Sayısı</label>
            <input
              type="number"
              className="form-input"
              min="50"
              max="500"
              value={trainParams.limit}
              onChange={(e) => setTrainParams({ ...trainParams, limit: Number(e.target.value) })}
            />
          </div>

          <div className="form-group">
            <label className="form-label">Model Tipi</label>
            <select
              className="form-select"
              value={trainParams.model_type}
              onChange={(e) => setTrainParams({ ...trainParams, model_type: e.target.value })}
            >
              <option value="random_forest">Random Forest (Hızlı)</option>
              <option value="gradient_boosting">Gradient Boosting (Yüksek Accuracy)</option>
              <option value="ensemble">Ensemble (En İyi)</option>
            </select>
          </div>

          <button
            className="btn btn-primary w-full"
            onClick={handleTrain}
            disabled={training}
          >
            {training ? (
              <>
                <Loader className="animate-spin" size={18} />
                Eğitiliyor...
              </>
            ) : (
              <>
                <Brain size={18} />
                Modeli Eğit
              </>
            )}
          </button>
        </div>

        {/* Generate with ML */}
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">ML ile Soru Üret</h3>
          </div>

          <div className="form-group">
            <label className="form-label">Konu (Opsiyonel)</label>
            <select
              className="form-select"
              value={generateParams.topic}
              onChange={(e) => setGenerateParams({ ...generateParams, topic: e.target.value })}
              disabled={loadingTopics}
            >
              <option value="">Karma Sorular</option>
              {topics.map((topic, index) => (
                <option key={index} value={topic}>
                  {topic}
                </option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label className="form-label">Soru Sayısı</label>
            <input
              type="number"
              className="form-input"
              min="1"
              max="20"
              value={generateParams.count}
              onChange={(e) => setGenerateParams({ ...generateParams, count: Number(e.target.value) })}
            />
          </div>

          <div className="form-group">
            <label className="form-label">Zorluk</label>
            <select
              className="form-select"
              value={generateParams.difficulty}
              onChange={(e) => setGenerateParams({ ...generateParams, difficulty: e.target.value })}
            >
              <option value="kolay">Kolay</option>
              <option value="orta">Orta</option>
              <option value="zor">Zor</option>
            </select>
          </div>

          <button
            className="btn btn-success w-full"
            onClick={handleGenerate}
            disabled={loading || !mlStatus?.is_trained}
          >
            {loading ? (
              <>
                <Loader className="animate-spin" size={18} />
                Üretiliyor...
              </>
            ) : (
              <>
                <Zap size={18} />
                Soru Üret
              </>
            )}
          </button>

          <button
            className="btn btn-secondary w-full mt-2"
            onClick={handleTrainAndGenerate}
            disabled={training || loading}
          >
            {training || loading ? (
              <>
                <Loader className="animate-spin" size={18} />
                İşleniyor...
              </>
            ) : (
              <>
                <Brain size={18} />
                Eğit + Üret (Tek Seferde)
              </>
            )}
          </button>
        </div>
      </div>

      {error && (
        <div className="alert alert-error mt-4">
          {error}
        </div>
      )}

      {success && (
        <div className="alert alert-success mt-4">
          {success}
        </div>
      )}

      {/* Generated Questions */}
      {questions.length > 0 && (
        <div className="card mt-4">
          <div className="card-header flex justify-between items-center">
            <h3 className="card-title">Üretilen Sorular ({questions.length})</h3>
            <div className="flex gap-2">
              {!showResults && Object.keys(userAnswers).length > 0 && (
                <button className="btn btn-primary" onClick={() => setShowResults(true)}>
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
              <button className="btn btn-success" onClick={handleDownloadPDF}>
                <FileDown size={18} />
                PDF İndir
              </button>
            </div>
          </div>

          {showResults && (
            <div className="exam-results">
              <h4>Sınav Sonucu</h4>
              <div className="score-display">
                <div className="score-item">
                  <span className="score-label">Doğru:</span>
                  <span className="score-value correct">
                    {questions.filter((q, i) => userAnswers[i] === q.correct_option).length}
                  </span>
                </div>
                <div className="score-item">
                  <span className="score-label">Yanlış:</span>
                  <span className="score-value wrong">
                    {questions.length - questions.filter((q, i) => userAnswers[i] === q.correct_option).length}
                  </span>
                </div>
                <div className="score-item">
                  <span className="score-label">Başarı:</span>
                  <span className="score-value percentage">
                    {((questions.filter((q, i) => userAnswers[i] === q.correct_option).length / questions.length) * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>
          )}

          <div className="generated-questions">
            {questions.map((q, index) => (
              <div key={index} className="generated-question">
                <div className="question-header">
                  <span className="question-number">Soru {index + 1}</span>
                  {q.ml_predicted_topic && (
                    <span className="badge badge-primary">
                      ML: {q.ml_predicted_topic} ({q.ml_confidence?.toFixed(1)}%)
                    </span>
                  )}
                </div>

                <div className="question-text">
                  <strong>Soru:</strong> {q.question_text}
                </div>

                <div className="options-grid">
                  {['A', 'B', 'C', 'D'].map(option => {
                    const optionText = q[`option_${option.toLowerCase()}`];
                    const isSelected = userAnswers[index] === option;
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
                        onClick={() => !showResults && setUserAnswers(prev => ({ ...prev, [index]: option }))}
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
                    {userAnswers[index] && (
                      <span className={`user-answer ${userAnswers[index] === q.correct_option ? 'correct' : 'wrong'}`}>
                        Sizin Cevabınız: <strong>{userAnswers[index]}</strong>
                        {userAnswers[index] === q.correct_option ? ' (Doğru)' : ' (Yanlış)'}
                      </span>
                    )}
                  </div>
                )}

                {showResults && q.explanation && (
                  <div className="explanation">
                    <strong>Açıklama:</strong> {q.explanation}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default MLModel;
