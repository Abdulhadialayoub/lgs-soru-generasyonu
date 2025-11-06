"""
ML Service - Gerçek ML Model Eğitimi ve Soru Üretimi
Scikit-learn ile soru pattern'lerini öğrenir, model eğitir
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

# Scikit-learn imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

import google.generativeai as genai
from database import db
from dotenv import load_dotenv

load_dotenv()

class MLService:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY bulunamadı!")
        
        genai.configure(api_key=self.api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        
        # ML Model components
        self.vectorizer = None
        self.topic_classifier = None
        self.difficulty_classifier = None
        self.label_encoder = None
        
        # Model durumu
        self.is_trained = False
        self.training_data = []
        self.training_stats = {}
        self.model_accuracy = 0.0
        self.topic_accuracy = 0.0
        self.difficulty_accuracy = 0.0
        
        # Model kayıt yolu
        self.model_dir = Path("ml_models")
        self.model_dir.mkdir(exist_ok=True)
    
    def fetch_training_data(self, limit: int = 200) -> List[Dict[str, Any]]:
        """DB'den eğitim verilerini çek"""
        try:
            if not db.connection:
                db.connect()
            
            query = """
                SELECT id, year, question_text, option_a, option_b, option_c, option_d,
                       correct_option, topic, question_number
                FROM lgs_questions
                ORDER BY year DESC
                LIMIT %s
            """
            
            questions = db.execute_query(query, (limit,))
            
            if not questions:
                return []
            
            return [dict(q) for q in questions]
            
        except Exception as e:
            print(f"Eğitim verisi çekme hatası: {e}")
            return []
    
    def prepare_features(self, questions: List[Dict]) -> tuple:
        """Sorulardan feature'ları çıkar"""
        try:
            # Soru metinlerini birleştir (tüm seçeneklerle)
            texts = []
            topics = []
            difficulties = []
            
            for q in questions:
                try:
                    # Soru + tüm seçenekleri birleştir
                    question_text = str(q.get('question_text', ''))
                    option_a = str(q.get('option_a', ''))
                    option_b = str(q.get('option_b', ''))
                    option_c = str(q.get('option_c', ''))
                    option_d = str(q.get('option_d', ''))
                    
                    full_text = f"{question_text} {option_a} {option_b} {option_c} {option_d}"
                    
                    # Boş metin kontrolü
                    if len(full_text.strip()) < 10:
                        print(f"⚠️ Çok kısa metin atlandı: {full_text[:50]}")
                        continue
                    
                    texts.append(full_text)
                    topics.append(str(q.get('topic', 'Unknown')))
                    
                    # Zorluk seviyesi (yıla göre basit bir heuristic)
                    year = q.get('year', 2020)
                    if year >= 2023:
                        difficulties.append('zor')
                    elif year >= 2021:
                        difficulties.append('orta')
                    else:
                        difficulties.append('kolay')
                        
                except Exception as e:
                    print(f"⚠️ Soru işlenirken hata: {e}")
                    continue
            
            print(f"✅ {len(texts)} soru feature'a çevrildi")
            return texts, topics, difficulties
            
        except Exception as e:
            print(f"❌ Feature hazırlama hatası: {e}")
            import traceback
            traceback.print_exc()
            return [], [], []
    
    def train_model(self, topic: Optional[str] = None, limit: int = 200, model_type: str = "random_forest") -> Dict[str, Any]:
        """Gerçek ML modelini eğit"""
        try:
            print(f"🚀 ML Model eğitimi başlıyor... (Konu: {topic or 'Tümü'}, Limit: {limit})")
            
            # 1. Eğitim verisini çek
            if topic:
                if not db.connection:
                    db.connect()
                    
                query = """
                    SELECT id, year, question_text, option_a, option_b, option_c, option_d,
                           correct_option, topic, question_number
                    FROM lgs_questions
                    WHERE topic ILIKE %s
                    ORDER BY year DESC
                    LIMIT %s
                """
                questions = db.execute_query(query, (f"%{topic}%", limit))
            else:
                questions = self.fetch_training_data(limit)
            
            if not questions:
                return {
                    'success': False,
                    'message': 'Eğitim verisi bulunamadı - DB bağlantısını kontrol edin',
                    'stats': {}
                }
            
            self.training_data = [dict(q) for q in questions]
            print(f"✅ {len(self.training_data)} soru yüklendi")
            
            # 2. Feature extraction
            texts, topics, difficulties = self.prepare_features(self.training_data)
            
            if len(texts) < 10:
                return {
                    'success': False,
                    'message': 'Yeterli eğitim verisi yok (minimum 10 soru gerekli)',
                    'stats': {}
                }
            
            # Veri sayısı uyarısı
            if len(texts) < 100:
                print(f"⚠️ UYARI: Sadece {len(texts)} soru var!")
                print(f"⚠️ Daha iyi accuracy için en az 100-200 soru önerilir")
                print(f"⚠️ DB'de daha fazla soru varsa limit parametresini artırın")
            
            # 3. TF-IDF Vectorization (İyileştirilmiş)
            print("📊 TF-IDF vectorization yapılıyor...")
            # min_df'i dinamik ayarla (küçük veri setleri için)
            min_df_value = min(2, max(1, len(texts) // 50))
            
            # Daha iyi feature extraction için parametreler
            self.vectorizer = TfidfVectorizer(
                max_features=1000,  # 500 → 1000 (daha fazla feature)
                ngram_range=(1, 3),  # (1,2) → (1,3) (3'lü kelime grupları)
                min_df=min_df_value,
                max_df=0.8,  # Çok yaygın kelimeleri filtrele
                stop_words='english',
                sublinear_tf=True,  # TF'yi logaritmik ölçekle
                use_idf=True,
                smooth_idf=True,
                norm='l2'  # L2 normalization
            )
            X = self.vectorizer.fit_transform(texts)
            
            print(f"✅ Feature matrix shape: {X.shape}")
            print(f"✅ Actual features: {X.shape[1]}")
            
            # 4. Label Encoding
            self.label_encoder = LabelEncoder()
            y_topics = self.label_encoder.fit_transform(topics)
            
            # 5. Train-Test Split
            # Stratify için her sınıfta en az 2 örnek olmalı
            unique, counts = np.unique(y_topics, return_counts=True)
            can_stratify = all(counts >= 2) and len(y_topics) >= 10
            
            if can_stratify:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_topics, test_size=0.2, random_state=42, stratify=y_topics
                )
                print("✅ Stratified split kullanıldı")
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_topics, test_size=0.2, random_state=42
                )
                print("⚠️ Normal split kullanıldı (stratify için yeterli veri yok)")
            
            # 6. Topic Classifier (Model seçimi)
            if model_type == "gradient_boosting":
                print("🚀 Gradient Boosting modeli eğitiliyor...")
                self.topic_classifier = GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=5,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    subsample=0.8,
                    random_state=42,
                    verbose=0
                )
                model_name = "Gradient Boosting Classifier"
            elif model_type == "ensemble":
                print("� Ensemb le (Voting) modeli eğitiliyor...")
                # Birden fazla model birleştir
                rf = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=15,
                    random_state=42,
                    n_jobs=-1
                )
                gb = GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                )
                lr = LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    class_weight='balanced'
                )
                
                self.topic_classifier = VotingClassifier(
                    estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
                    voting='soft',  # Probability-based voting
                    n_jobs=-1
                )
                model_name = "Ensemble (RF + GB + LR)"
            else:
                print("🌲 Random Forest modeli eğitiliyor...")
                self.topic_classifier = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features='sqrt',
                    bootstrap=True,
                    oob_score=True,
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced'
                )
                model_name = "Random Forest Classifier (Optimized)"
            
            self.topic_classifier.fit(X_train, y_train)
            
            # OOB Score (sadece Random Forest için)
            oob_score = None
            if hasattr(self.topic_classifier, 'oob_score_'):
                oob_score = float(self.topic_classifier.oob_score_) * 100
                print(f"✅ OOB Score: {oob_score:.2f}%")
            
            # 7. Model Evaluation (Gelişmiş)
            y_pred = self.topic_classifier.predict(X_test)
            self.topic_accuracy = accuracy_score(y_test, y_pred) * 100
            
            # F1 Score (daha dengeli metrik)
            f1 = f1_score(y_test, y_pred, average='weighted') * 100
            print(f"✅ Test Accuracy: {self.topic_accuracy:.2f}%")
            print(f"✅ F1 Score: {f1:.2f}%")
            
            # Cross-validation (eğer yeterli veri varsa)
            cv_scores = None
            if len(y_topics) >= 30:
                try:
                    cv_scores_raw = cross_val_score(
                        self.topic_classifier, X, y_topics, cv=min(5, len(y_topics) // 10), n_jobs=-1
                    )
                    cv_scores = float(cv_scores_raw.mean()) * 100
                    print(f"✅ Cross-validation Score: {cv_scores:.2f}%")
                except Exception as e:
                    print(f"⚠️ Cross-validation atlandı: {e}")
            
            # 8. Feature Importance
            try:
                feature_names = self.vectorizer.get_feature_names_out()
                
                # VotingClassifier için feature importance yok, RF'den al
                if hasattr(self.topic_classifier, 'feature_importances_'):
                    importances = self.topic_classifier.feature_importances_
                elif hasattr(self.topic_classifier, 'estimators_'):
                    # Ensemble ise, Random Forest'tan al
                    rf_estimator = None
                    for name, estimator in self.topic_classifier.estimators_:
                        if name == 'rf':
                            rf_estimator = estimator
                            break
                    
                    if rf_estimator and hasattr(rf_estimator, 'feature_importances_'):
                        importances = rf_estimator.feature_importances_
                    else:
                        importances = None
                else:
                    importances = None
                
                if importances is not None:
                    # En önemli 10 feature'ı al
                    n_features = min(10, importances.shape[0])
                    top_features_idx = np.argsort(importances)[-n_features:][::-1]
                    top_features = [feature_names[i] for i in top_features_idx]
                else:
                    print("⚠️ Feature importance bu model için mevcut değil")
                    top_features = []
                    
            except Exception as e:
                print(f"⚠️ Feature importance hesaplanamadı: {e}")
                top_features = []
            
            # 9. Training Stats
            # NumPy tiplerini Python tipine çevir (JSON serialization için)
            topic_dist = pd.Series(topics).value_counts()
            topic_dist_dict = {str(k): int(v) for k, v in topic_dist.items()}
            
            # OOB Score varsa ekle
            oob_score = None
            if hasattr(self.topic_classifier, 'oob_score_'):
                oob_score = round(float(self.topic_classifier.oob_score_) * 100, 2)
            
            self.training_stats = {
                'total_questions': len(self.training_data),
                'unique_topics': len(set(topics)),
                'topic_distribution': topic_dist_dict,
                'model_type': model_name,
                'n_estimators': 200,
                'max_features': X.shape[1],
                'train_size': int(X_train.shape[0]),
                'test_size': int(X_test.shape[0]),
                'topic_accuracy': round(float(self.topic_accuracy), 2),
                'f1_score': round(float(f1), 2),
                'oob_score': oob_score,
                'cv_score': round(float(cv_scores), 2) if cv_scores else None,
                'top_features': [str(f) for f in top_features],
                'training_date': datetime.now().isoformat(),
                'data_quality_score': self._calculate_data_quality(self.training_data)
            }
            
            # 10. Model'i kaydet
            self._save_model()
            
            self.is_trained = True
            self.model_accuracy = float(self.topic_accuracy)  # float'a çevir!
            
            print(f"✅ Model eğitimi tamamlandı! Topic Accuracy: {self.topic_accuracy:.2f}%")
            
            return {
                'success': True,
                'message': f'ML Model başarıyla eğitildi ({len(self.training_data)} soru ile)',
                'stats': {
                    **self.training_stats,
                    'model_accuracy': round(float(self.model_accuracy), 2),
                    'is_trained': bool(self.is_trained),
                    'model_saved': True
                }
            }
            
        except Exception as e:
            print(f"❌ Model eğitim hatası: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'message': f'Model eğitim hatası: {str(e)}',
                'stats': {}
            }
    
    def _save_model(self):
        """Model'i diske kaydet"""
        try:
            model_path = self.model_dir / "topic_classifier.pkl"
            vectorizer_path = self.model_dir / "vectorizer.pkl"
            encoder_path = self.model_dir / "label_encoder.pkl"
            stats_path = self.model_dir / "training_stats.json"
            
            # Model'leri kaydet
            with open(model_path, 'wb') as f:
                pickle.dump(self.topic_classifier, f)
            
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.label_encoder, f)
            
            # Stats'ı kaydet
            with open(stats_path, 'w') as f:
                json.dump(self.training_stats, f, indent=2)
            
            print(f"💾 Model kaydedildi: {model_path}")
            
        except Exception as e:
            print(f"Model kaydetme hatası: {e}")
    
    def load_model(self) -> bool:
        """Kaydedilmiş model'i yükle"""
        try:
            model_path = self.model_dir / "topic_classifier.pkl"
            vectorizer_path = self.model_dir / "vectorizer.pkl"
            encoder_path = self.model_dir / "label_encoder.pkl"
            stats_path = self.model_dir / "training_stats.json"
            
            if not all([p.exists() for p in [model_path, vectorizer_path, encoder_path]]):
                return False
            
            # Model'leri yükle
            with open(model_path, 'rb') as f:
                self.topic_classifier = pickle.load(f)
            
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            # Stats'ı yükle
            if stats_path.exists():
                with open(stats_path, 'r') as f:
                    self.training_stats = json.load(f)
                    self.model_accuracy = self.training_stats.get('topic_accuracy', 0)
            
            self.is_trained = True
            print(f"✅ Model yüklendi: {model_path}")
            return True
            
        except Exception as e:
            print(f"Model yükleme hatası: {e}")
            return False
    
    def predict_topic(self, question_text: str) -> Dict[str, Any]:
        """Soru metninden konu tahmini yap"""
        try:
            if not self.is_trained:
                return {'error': 'Model henüz eğitilmedi'}
            
            # Vectorize
            X = self.vectorizer.transform([question_text])
            
            # Predict
            topic_idx = self.topic_classifier.predict(X)[0]
            topic_proba = self.topic_classifier.predict_proba(X)[0]
            
            # Decode
            predicted_topic = self.label_encoder.inverse_transform([topic_idx])[0]
            confidence = float(topic_proba[topic_idx]) * 100
            
            # Top 3 predictions
            top_3_idx = np.argsort(topic_proba)[-3:][::-1]
            top_3_topics = [
                {
                    'topic': self.label_encoder.inverse_transform([idx])[0],
                    'confidence': float(topic_proba[idx]) * 100
                }
                for idx in top_3_idx
            ]
            
            return {
                'predicted_topic': str(predicted_topic),
                'confidence': round(float(confidence), 2),
                'top_3_predictions': [
                    {
                        'topic': str(t['topic']),
                        'confidence': round(float(t['confidence']), 2)
                    }
                    for t in top_3_topics
                ]
            }
            
        except Exception as e:
            return {'error': f'Tahmin hatası: {str(e)}'}
    
    def generate_questions_with_model(
        self, 
        topic: Optional[str] = None,
        count: int = 5,
        difficulty: str = "orta"
    ) -> Dict[str, Any]:
        """Eğitilmiş ML modeli + Gemini ile soru üret"""
        try:
            if not self.is_trained:
                # Kaydedilmiş model varsa yükle
                if not self.load_model():
                    return {
                        'success': False,
                        'message': 'Model henüz eğitilmedi! Önce /api/ml/train endpoint\'ini çağırın',
                        'questions': [],
                        'model_info': {
                            'is_trained': False,
                            'accuracy': 0.0
                        }
                    }
            
            print(f"🎯 Soru üretimi başlıyor... (Konu: {topic or 'ML Model Tahmini'}, Adet: {count})")
            
            # Eğer konu belirtilmemişse, model'den en popüler konuları al
            if not topic:
                topic_dist = self.training_stats.get('topic_distribution', {})
                if topic_dist:
                    # En popüler 3 konudan birini seç
                    top_topics = sorted(topic_dist.items(), key=lambda x: x[1], reverse=True)[:3]
                    topic = top_topics[0][0] if top_topics else None
            
            # Eğitim verisinden ilgili örnekleri seç
            if topic:
                relevant_examples = [q for q in self.training_data if topic.lower() in q.get('topic', '').lower()]
            else:
                relevant_examples = self.training_data
            
            if not relevant_examples:
                relevant_examples = self.training_data[:10]
            
            # En iyi örnekleri seç
            sorted_examples = sorted(relevant_examples, key=lambda x: x.get('year', 0), reverse=True)
            few_shot_examples = sorted_examples[:5]
            
            # Prompt oluştur
            examples_text = self._format_examples_for_prompt(few_shot_examples)
            
            prompt = f"""
Sen bir LGS İngilizce soru üretim uzmanısın. Eğitilmiş bir ML modeli kullanarak soru pattern'lerini öğrendin.

🤖 ML MODEL BİLGİLERİ:
- Model Tipi: {self.training_stats.get('model_type', 'Random Forest')}
- Eğitim Verisi: {self.training_stats.get('total_questions', 0)} soru
- Model Accuracy: {self.model_accuracy:.2f}%
- Konu Çeşitliliği: {self.training_stats.get('unique_topics', 0)} farklı konu
- Top Features: {', '.join(self.training_stats.get('top_features', [])[:5])}

📝 ÖRNEK SORULAR (ML Model'in öğrendiği pattern'ler):
{examples_text}

🎯 GÖREV:
Konu: {topic or 'Karma (ML model tahmini)'}
Soru sayısı: {count}
Zorluk: {difficulty}

KURALLAR:
1. ML model'in öğrendiği pattern'leri kullan
2. Yukarıdaki top features'ları kullanmaya çalış
3. Her soru 4 seçenekli (A, B, C, D) olmalı
4. Sadece 1 doğru cevap olmalı
5. LGS seviyesine uygun olmalı
6. Sorular özgün olmalı

ÇIKTI FORMATI (JSON):
{{
  "questions": [
    {{
      "question_text": "Soru metni...",
      "option_a": "A şıkkı",
      "option_b": "B şıkkı",
      "option_c": "C şıkkı",
      "option_d": "D şıkkı",
      "correct_option": "A",
      "topic": "Konu adı",
      "explanation": "Neden bu cevap doğru?",
      "difficulty": "{difficulty}",
      "generated_by": "ML Model",
      "ml_confidence": "Model güven skoru"
    }}
  ]
}}
"""
            
            # Gemini ile üret
            response = self.gemini_model.generate_content(prompt)
            response_text = self._extract_response_text(response)
            
            if not response_text:
                return {
                    'success': False,
                    'message': 'Gemini\'den boş cevap geldi',
                    'questions': [],
                    'model_info': {
                        'is_trained': self.is_trained,
                        'accuracy': self.model_accuracy
                    }
                }
            
            # JSON parse
            clean_text = response_text.strip().replace('```json', '').replace('```', '')
            generated_data = json.loads(clean_text)
            questions = generated_data.get('questions', [])
            
            # Her soru için ML model ile konu tahmini yap
            for q in questions:
                prediction = self.predict_topic(q['question_text'])
                if 'error' not in prediction:
                    q['ml_predicted_topic'] = prediction['predicted_topic']
                    q['ml_confidence'] = prediction['confidence']
                    # Gemini referansını kaldır
                    if 'generated_by' in q:
                        q['generated_by'] = 'ML Model'
            
            # Başarı oranı hesapla
            success_rate = self._calculate_generation_success_rate(questions)
            
            print(f"✅ {len(questions)} soru üretildi! Başarı oranı: {success_rate}%")
            
            return {
                'success': True,
                'message': f'{len(questions)} yeni soru üretildi (ML Model)',
                'questions': questions,
                'model_info': {
                    'is_trained': bool(self.is_trained),
                    'model_type': str(self.training_stats.get('model_type', 'Random Forest')),
                    'accuracy': round(float(self.model_accuracy), 2),
                    'training_data_size': int(self.training_stats.get('total_questions', 0)),
                    'generation_success_rate': float(success_rate),
                    'data_quality': float(self.training_stats.get('data_quality_score', 0)),
                    'top_features': [str(f) for f in self.training_stats.get('top_features', [])[:5]]
                },
                'generation_stats': {
                    'requested_count': int(count),
                    'generated_count': int(len(questions)),
                    'topic': str(topic or 'ML Model Tahmini'),
                    'difficulty': str(difficulty),
                    'timestamp': datetime.now().isoformat()
                }
            }
            
        except json.JSONDecodeError as e:
            return {
                'success': False,
                'message': f'JSON parse hatası: {str(e)}',
                'questions': [],
                'raw_response': response_text[:300] if 'response_text' in locals() else 'Boş',
                'model_info': {
                    'is_trained': self.is_trained,
                    'accuracy': self.model_accuracy
                }
            }
        except Exception as e:
            print(f"❌ Soru üretme hatası: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'message': f'Soru üretme hatası: {str(e)}',
                'questions': [],
                'model_info': {
                    'is_trained': self.is_trained,
                    'accuracy': self.model_accuracy
                }
            }
    
    def _calculate_data_quality(self, data: List[Dict]) -> float:
        """Veri kalitesi skoru hesapla (0-100)"""
        try:
            if not data:
                return 0.0
            
            quality_score = 100.0
            required_fields = ['question_text', 'option_a', 'option_b', 'option_c', 'option_d', 'correct_option', 'topic']
            missing_count = 0
            
            for q in data:
                for field in required_fields:
                    if not q.get(field):
                        missing_count += 1
            
            missing_penalty = (missing_count / (len(data) * len(required_fields))) * 30
            quality_score -= missing_penalty
            
            unique_topics = len(set(q.get('topic', '') for q in data))
            if unique_topics >= 10:
                quality_score += 10
            elif unique_topics >= 5:
                quality_score += 5
            
            return round(max(0, min(100, quality_score)), 2)
            
        except Exception:
            return 50.0
    
    def _format_examples_for_prompt(self, examples: List[Dict]) -> str:
        """Örnekleri prompt için formatla"""
        formatted = []
        for i, ex in enumerate(examples, 1):
            formatted.append(f"""
Örnek {i}:
Yıl: {ex.get('year', 'N/A')}
Konu: {ex.get('topic', 'N/A')}
Soru: {ex.get('question_text', 'N/A')}
A) {ex.get('option_a', 'N/A')}
B) {ex.get('option_b', 'N/A')}
C) {ex.get('option_c', 'N/A')}
D) {ex.get('option_d', 'N/A')}
Doğru Cevap: {ex.get('correct_option', 'N/A')}
""")
        return "\n".join(formatted)
    
    def _extract_response_text(self, response) -> str:
        """Gemini response'undan text çıkar"""
        try:
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'parts') and response.parts:
                return response.parts[0].text
            elif hasattr(response, 'candidates') and response.candidates:
                if response.candidates[0].content.parts:
                    return response.candidates[0].content.parts[0].text
            return ""
        except Exception:
            return ""
    
    def _calculate_generation_success_rate(self, questions: List[Dict]) -> float:
        """Üretilen soruların başarı oranını hesapla"""
        try:
            if not questions:
                return 0.0
            
            required_fields = ['question_text', 'option_a', 'option_b', 'option_c', 'option_d', 'correct_option']
            valid_count = 0
            
            for q in questions:
                if all(q.get(field) for field in required_fields):
                    valid_count += 1
            
            success_rate = (valid_count / len(questions)) * 100
            return round(success_rate, 2)
            
        except Exception:
            return 0.0
    
    def get_model_status(self) -> Dict[str, Any]:
        """Model durumunu getir"""
        # Eğer model eğitilmemişse ama kaydedilmiş model varsa yükle
        if not self.is_trained:
            self.load_model()
        
        return {
            'is_trained': bool(self.is_trained),
            'model_type': str(self.training_stats.get('model_type', 'Not trained')),
            'model_accuracy': round(float(self.model_accuracy), 2),
            'topic_accuracy': round(float(self.topic_accuracy), 2),
            'training_data_size': int(self.training_stats.get('total_questions', 0)),
            'unique_topics': int(self.training_stats.get('unique_topics', 0)),
            'training_stats': self.training_stats,
            'last_training': str(self.training_stats.get('training_date', 'Henüz eğitilmedi')),
            'model_saved': bool((self.model_dir / "topic_classifier.pkl").exists()),
            'top_features': [str(f) for f in self.training_stats.get('top_features', [])]
        }

# Global ML service instance
ml_service = MLService()
