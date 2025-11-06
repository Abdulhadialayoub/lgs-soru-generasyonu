import { useState } from 'react';
import { apiService } from '../config/api';
import { FileText, Loader, Download, FileDown } from 'lucide-react';
import jsPDF from 'jspdf';
import './CreateExam.css';

function CreateExam() {
  const [questionCount, setQuestionCount] = useState(10);
  const [exam, setExam] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [examMode, setExamMode] = useState(true); // Varsayılan olarak açık
  const [userAnswers, setUserAnswers] = useState({});
  const [showResults, setShowResults] = useState(false);

  const handleGenerateExam = async () => {
    try {
      setLoading(true);
      setError('');
      
      console.log('Sınav oluşturuluyor, soru sayısı:', questionCount);
      const response = await apiService.generateExam(questionCount);
      console.log('Backend yanıtı:', response.data);
      
      if (response.data.success) {
        setExam(response.data.data);
      } else {
        setError(response.data.message);
      }
    } catch (err) {
      setError('Sınav oluşturulurken hata: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadPDF = () => {
    if (!exam) return;

    const doc = new jsPDF();
    
    // Türkçe karakter desteği için font ayarı
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
    doc.text('LGS INGILIZCE SINAVI', pageWidth / 2, yPosition, { align: 'center' });
    yPosition += 15;

    // Sınav Bilgileri
    doc.setFontSize(10);
    doc.setFont('helvetica', 'normal');
    doc.text(`Toplam Soru: ${exam.exam_info.total_questions}`, margin, yPosition);
    yPosition += 7;
    doc.text(`Zorluk: ${exam.exam_info.difficulty}`, margin, yPosition);
    yPosition += 7;
    doc.text(`Tahmini Sure: ${exam.exam_info.estimated_time}`, margin, yPosition);
    yPosition += 10;

    // Konu Dağılımı
    doc.setFont('helvetica', 'bold');
    doc.text('Konu Dagilimi:', margin, yPosition);
    yPosition += 7;
    doc.setFont('helvetica', 'normal');
    Object.entries(exam.exam_info.topic_distribution).forEach(([topic, count]) => {
      doc.text(`  - ${topic}: ${count} soru`, margin, yPosition);
      yPosition += 6;
    });
    yPosition += 10;

    // Sorular
    exam.questions.forEach((q, index) => {
      // Yeni sayfa kontrolü
      if (yPosition > pageHeight - 60) {
        doc.addPage();
        yPosition = margin;
      }

      // Soru numarası ve konu
      doc.setFontSize(12);
      doc.setFont('helvetica', 'bold');
      doc.text(`Soru ${index + 1}`, margin, yPosition);
      doc.setFontSize(9);
      doc.setFont('helvetica', 'italic');
      doc.text(`(${cleanText(q.exam_topic)})`, margin + 20, yPosition);
      yPosition += 8;

      // Soru metni
      doc.setFontSize(10);
      doc.setFont('helvetica', 'normal');
      const questionLines = doc.splitTextToSize(cleanText(q.question_text), maxWidth);
      doc.text(questionLines, margin, yPosition);
      yPosition += questionLines.length * 6 + 5;

      // Seçenekler
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
    const answersPerColumn = Math.ceil(exam.questions.length / 4);
    const columnWidth = (pageWidth - 2 * margin) / 4;
    
    exam.questions.forEach((q, index) => {
      const column = Math.floor(index / answersPerColumn);
      const row = index % answersPerColumn;
      const xPos = margin + (column * columnWidth);
      const yPos = yPosition + (row * 7);

      if (yPos > pageHeight - 30) return; // Sayfa taşmasını önle

      doc.text(`${index + 1}. ${q.correct_option}`, xPos, yPos);
    });

    yPosition += (answersPerColumn * 7) + 15;

    // İstatistikler
    if (yPosition < pageHeight - 40) {
      doc.setFont('helvetica', 'bold');
      doc.text('Konu Bazinda Cevap Dagilimi:', margin, yPosition);
      yPosition += 10;
      doc.setFont('helvetica', 'normal');

      Object.entries(exam.exam_info.topic_distribution).forEach(([topic, count]) => {
        if (yPosition > pageHeight - 20) return;
        const topicAnswers = exam.questions
          .filter(q => q.exam_topic === topic)
          .map((q, i) => `${exam.questions.indexOf(q) + 1}.${q.correct_option}`)
          .join(', ');
        
        const line = `${topic} (${count} soru): ${topicAnswers}`;
        const lines = doc.splitTextToSize(line, maxWidth);
        doc.text(lines, margin, yPosition);
        yPosition += lines.length * 6 + 3;
      });
    }

    // PDF'i kaydet
    const fileName = `LGS_Sinav_${new Date().toISOString().split('T')[0]}.pdf`;
    doc.save(fileName);
  };

  return (
    <div className="create-exam">
      <div className="page-header">
        <h1 className="page-title">Sınav Oluştur</h1>
        <p className="page-subtitle">Gerçekçi LGS İngilizce sınavı oluşturun</p>
      </div>

      <div className="exam-layout">
        {/* Form */}
        <div className="card exam-form">
          <div className="card-header">
            <h3 className="card-title">Sınav Ayarları</h3>
          </div>

          <div className="form-group">
            <label className="form-label">Soru Sayısı</label>
            <input
              type="number"
              className="form-input"
              min="5"
              max="20"
              value={questionCount}
              onChange={(e) => setQuestionCount(Number(e.target.value))}
            />
            <small className="form-hint">5-20 arası soru seçebilirsiniz</small>
          </div>

          <button
            className="btn btn-primary w-full"
            onClick={handleGenerateExam}
            disabled={loading}
          >
            {loading ? (
              <>
                <Loader className="animate-spin" size={18} />
                Oluşturuluyor...
              </>
            ) : (
              <>
                <FileText size={18} />
                Sınav Oluştur
              </>
            )}
          </button>

          {error && (
            <div className="alert alert-error mt-3">
              {error}
            </div>
          )}

          {exam && (
            <>
              <div className="exam-info mt-4">
                <h4 className="info-title">Sınav Bilgileri</h4>
                <div className="info-item">
                  <span>Toplam Soru:</span>
                  <strong>{exam.exam_info.total_questions}</strong>
                </div>
                <div className="info-item">
                  <span>Zorluk:</span>
                  <strong>{exam.exam_info.difficulty}</strong>
                </div>
                <div className="info-item">
                  <span>Tahmini Süre:</span>
                  <strong>{exam.exam_info.estimated_time}</strong>
                </div>
                <div className="info-item">
                  <span>Sınav Tipi:</span>
                  <strong>{exam.exam_info.exam_type}</strong>
                </div>
              </div>

              <div className="topic-distribution mt-3">
                <h4 className="info-title">Konu Dağılımı</h4>
                {Object.entries(exam.exam_info.topic_distribution).map(([topic, count]) => (
                  <div key={topic} className="distribution-item">
                    <span>{topic}</span>
                    <span className="badge badge-primary">{count} soru</span>
                  </div>
                ))}
              </div>

              <button
                className="btn btn-success w-full mt-3"
                onClick={handleDownloadPDF}
              >
                <FileDown size={18} />
                Sınavı PDF Olarak İndir
              </button>
            </>
          )}
        </div>

        {/* Exam Preview */}
        <div className="card exam-preview-container">
          <div className="card-header flex justify-between items-center">
            <h3 className="card-title">Sınav Önizleme</h3>
            {exam && (
              <div className="flex gap-2">
                {!showResults && Object.keys(userAnswers).length > 0 && (
                  <button className="btn btn-primary" onClick={() => setShowResults(true)}>
                    Cevapları Kontrol Et ({Object.keys(userAnswers).length}/{exam.questions.length})
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
            )}
          </div>

          {showResults && exam && (
            <div className="exam-results">
              <h4>Sınav Sonucu</h4>
              <div className="score-display">
                <div className="score-item">
                  <span className="score-label">Doğru:</span>
                  <span className="score-value correct">
                    {exam.questions.filter((q, i) => userAnswers[i] === q.correct_option).length}
                  </span>
                </div>
                <div className="score-item">
                  <span className="score-label">Yanlış:</span>
                  <span className="score-value wrong">
                    {exam.questions.length - exam.questions.filter((q, i) => userAnswers[i] === q.correct_option).length}
                  </span>
                </div>
                <div className="score-item">
                  <span className="score-label">Başarı:</span>
                  <span className="score-value percentage">
                    {((exam.questions.filter((q, i) => userAnswers[i] === q.correct_option).length / exam.questions.length) * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>
          )}

          {loading ? (
            <div className="loading">
              <div className="spinner"></div>
              <p className="mt-2">Sınav oluşturuluyor...</p>
            </div>
          ) : exam ? (
            <div className="exam-preview">
              {exam.questions.map((q, index) => (
                <div key={index} className="exam-question">
                  <div className="question-header">
                    <span className="question-number">Soru {index + 1}</span>
                    <span className="badge badge-primary">{q.exam_topic}</span>
                  </div>

                  <div className="question-text">
                    {q.question_text}
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
              <FileText size={48} color="#cbd5e1" />
              <p>Henüz sınav oluşturulmadı</p>
              <p className="text-sm">Sol taraftaki ayarları yapıp "Sınav Oluştur" butonuna tıklayın</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default CreateExam;
