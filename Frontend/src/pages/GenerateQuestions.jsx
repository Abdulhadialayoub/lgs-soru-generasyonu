import { useState, useEffect } from 'react';
import { apiService } from '../config/api';
import { Sparkles, Loader, FileDown } from 'lucide-react';
import jsPDF from 'jspdf';
import './GenerateQuestions.css';

function GenerateQuestions() {
  const [formData, setFormData] = useState({
    topic: '',
    count: 5,
    difficulty: 'orta'
  });
  const [questions, setQuestions] = useState([]);
  const [topics, setTopics] = useState([]);
  const [loading, setLoading] = useState(false);
  const [loadingTopics, setLoadingTopics] = useState(true);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [examMode, setExamMode] = useState(true); // Varsayılan olarak açık
  const [userAnswers, setUserAnswers] = useState({});
  const [showResults, setShowResults] = useState(false);

  useEffect(() => {
    loadTopics();
  }, []);

  const loadTopics = async () => {
    try {
      setLoadingTopics(true);
      // Önce statistics'ten dene
      const statsResponse = await apiService.getStatisticsSummary();
      if (statsResponse.data.success && statsResponse.data.data.topic_summary) {
        const topicList = statsResponse.data.data.topic_summary || [];
        const popularTopics = topicList.slice(0, 20).map(t => t.topic);
        setTopics(popularTopics);
      } else {
        // Statistics yoksa questions'tan çek
        const questionsResponse = await apiService.getQuestions({ limit: 200 });
        if (questionsResponse.data.success) {
          const questions = questionsResponse.data.data || [];
          // Benzersiz konuları al
          const uniqueTopics = [...new Set(questions.map(q => q.topic))].filter(Boolean);
          setTopics(uniqueTopics.sort());
        }
      }
    } catch (err) {
      console.error('Konular yüklenemedi:', err);
      // Fallback: Sabit konu listesi
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

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleGenerate = async () => {
    try {
      setLoading(true);
      setError('');
      setSuccess('');
      
      const response = await apiService.generateQuestions(formData);
      
      if (response.data.success) {
        setQuestions(response.data.data.questions);
        setSuccess(response.data.message);
      } else {
        setError(response.data.message);
      }
    } catch (err) {
      setError('Soru üretilirken hata oluştu: ' + err.message);
    } finally {
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
    doc.text('MCP ILE URETILEN SORULAR', pageWidth / 2, yPosition, { align: 'center' });
    yPosition += 15;

    // Bilgiler
    doc.setFontSize(10);
    doc.setFont('helvetica', 'normal');
    doc.text(`Konu: ${formData.topic || 'Karma'}`, margin, yPosition);
    yPosition += 7;
    doc.text(`Zorluk: ${formData.difficulty}`, margin, yPosition);
    yPosition += 7;
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
      if (q.topic) {
        doc.setFontSize(9);
        doc.setFont('helvetica', 'italic');
        doc.text(`(${cleanText(q.topic)})`, margin + 20, yPosition);
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

    doc.save(`MCP_Sorular_${new Date().toISOString().split('T')[0]}.pdf`);
  };

  return (
    <div className="generate-questions">
      <div className="page-header">
        <h1 className="page-title">MCP Soru Üret</h1>
        <p className="page-subtitle">Model Context Protocol ile yeni sorular oluşturun</p>
      </div>

      <div className="exam-layout">
        {/* Form */}
        <div className="card exam-form">
          <div className="card-header">
            <h3 className="card-title">Soru Parametreleri</h3>
          </div>

          <div className="form-group">
            <label className="form-label">Konu (Opsiyonel)</label>
            <select
              name="topic"
              className="form-select"
              value={formData.topic}
              onChange={handleChange}
              disabled={loadingTopics}
            >
              <option value="">Karma (Tüm Konular)</option>
              {topics.map((topic, index) => (
                <option key={index} value={topic}>
                  {topic}
                </option>
              ))}
            </select>
            <small className="form-hint">
              {loadingTopics ? 'Konular yükleniyor...' : 'Boş bırakırsanız karma sorular üretilir'}
            </small>
          </div>

          <div className="form-group">
            <label className="form-label">Soru Sayısı</label>
            <input
              type="number"
              name="count"
              className="form-input"
              min="1"
              max="10"
              value={formData.count}
              onChange={handleChange}
            />
          </div>

          <div className="form-group">
            <label className="form-label">Zorluk Seviyesi</label>
            <select
              name="difficulty"
              className="form-select"
              value={formData.difficulty}
              onChange={handleChange}
            >
              <option value="kolay">Kolay</option>
              <option value="orta">Orta</option>
              <option value="zor">Zor</option>
            </select>
          </div>

          <button
            className="btn btn-primary w-full"
            onClick={handleGenerate}
            disabled={loading}
          >
            {loading ? (
              <>
                <Loader className="animate-spin" size={18} />
                Üretiliyor...
              </>
            ) : (
              <>
                <Sparkles size={18} />
                Soru Üret
              </>
            )}
          </button>

          {error && (
            <div className="alert alert-error mt-3">
              {error}
            </div>
          )}

          {success && (
            <div className="alert alert-success mt-3">
              {success}
            </div>
          )}
        </div>

        {/* Results */}
        <div className="card exam-preview-container">
          <div className="card-header flex justify-between items-center">
            <h3 className="card-title">Üretilen Sorular ({questions.length})</h3>
            {questions.length > 0 && (
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
            )}
          </div>

          {showResults && questions.length > 0 && (
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

          {loading ? (
            <div className="loading">
              <div className="spinner"></div>
              <p className="mt-2">Sorular üretiliyor...</p>
            </div>
          ) : questions.length > 0 ? (
            <div className="generated-questions">
              {questions.map((q, index) => (
                <div key={index} className="generated-question">
                  <div className="question-header">
                    <span className="question-number">Soru {index + 1}</span>
                    {q.topic && <span className="badge badge-primary">{q.topic}</span>}
                    {q.difficulty && <span className="badge badge-warning">{q.difficulty}</span>}
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
          ) : (
            <div className="empty-state">
              <Sparkles size={48} color="#cbd5e1" />
              <p>Henüz soru üretilmedi</p>
              <p className="text-sm">Sol taraftaki formu doldurup "Soru Üret" butonuna tıklayın</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default GenerateQuestions;
