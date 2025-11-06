from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from database import db
from gemini_service import gemini_service
from ml_service import ml_service
from typing import Optional
import uvicorn
import numpy as np

app = FastAPI(
    title="LGS Soru Tahmin API",
    description="AI Destekli LGS İngilizce Soru Tahmin Sistemi",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Uygulama başlatıldığında veritabanı bağlantısını kur"""
    if not db.connect():
        raise Exception("Veritabanı bağlantısı kurulamadı!")

@app.on_event("shutdown")
async def shutdown_event():
    """Uygulama kapatıldığında veritabanı bağlantısını kapat"""
    db.disconnect()

@app.get("/")
def read_root():
    """Ana sayfa"""
    return {
        "message": "LGS Soru Tahmin API'sine Hoş Geldiniz!",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/api/questions")
def get_questions(
    year: Optional[int] = Query(None, description="Yıl filtresi"),
    topic: Optional[str] = Query(None, description="Konu filtresi"),
    limit: Optional[int] = Query(50, description="Soru sayısı sınırı")
):
    """
    Geçmiş çıkmış soruları getir
    
    - **year**: Belirli yıl (opsiyonel)
    - **topic**: Belirli konu (opsiyonel)
    - **limit**: Maksimum soru sayısı (varsayılan: 50)
    """
    try:
        # Database bağlantısını kontrol et
        if not db.connection:
            db.connect()
        
        # Base query
        query = "SELECT * FROM lgs_questions WHERE 1=1"
        params = []
        
        # Filters
        if year:
            query += " AND year = %s"
            params.append(year)
        
        if topic:
            query += " AND topic ILIKE %s"
            params.append(f"%{topic}%")
        
        # Order and limit
        query += " ORDER BY year DESC, question_number ASC LIMIT %s"
        params.append(limit)
        
        # Execute
        questions = db.execute_query(query, params)
        
        # Null check
        if questions is None:
            return {
                "success": False,
                "message": "Veritabanı sorgusu başarısız",
                "data": [],
                "count": 0
            }
        
        if not questions:
            return {
                "success": True,
                "message": "Soru bulunamadı",
                "data": [],
                "count": 0
            }
        
        questions_list = [dict(q) for q in questions]
        
        return {
            "success": True,
            "message": f"{len(questions_list)} soru bulundu",
            "data": questions_list,
            "count": len(questions_list)
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Hata: {str(e)}",
            "data": [],
            "count": 0
        }

@app.post("/api/predict")
def predict_questions(target_year: Optional[int] = Query(2025, description="Tahmin yılı")):
    """
    Soru dağılımı tahmini
    
    - **target_year**: Hangi yıl için tahmin (varsayılan: 2025)
    """
    try:
        prediction = gemini_service.predict_question_distribution(target_year)
        
        return {
            "success": True,
            "message": f"{target_year} yılı tahmini",
            "data": prediction
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Tahmin hatası: {str(e)}",
            "data": {}
        }

@app.post("/api/generate")
def generate_questions(
    topic: Optional[str] = Query(None, description="Belirli konu (opsiyonel)"),
    count: Optional[int] = Query(5, description="Soru sayısı", ge=1, le=10),
    difficulty: Optional[str] = Query("orta", description="Zorluk: kolay, orta, zor")
):
    """
    Soru üretme
    
    - **topic**: Belirli konu için soru üret (boş bırakırsan genel)
    - **count**: Kaç soru üretilecek (1-10)
    - **difficulty**: Zorluk seviyesi
    """
    try:
        if difficulty not in ["kolay", "orta", "zor"]:
            return {
                "success": False,
                "message": "Zorluk 'kolay', 'orta' veya 'zor' olmalı",
                "data": {}
            }
        
        if topic:
            # Belirli konuda soru üret
            questions = gemini_service.generate_questions(topic, count, difficulty)
            message = f"{topic} konusunda {len(questions)} soru üretildi"
        else:
            # Genel karma soru üret
            questions = gemini_service.generate_mixed_questions(count)
            message = f"Karma {len(questions)} soru üretildi"
        
        return {
            "success": True,
            "message": message,
            "data": {
                "topic": topic or "Karma",
                "difficulty": difficulty,
                "count": len(questions),
                "questions": questions
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Soru üretme hatası: {str(e)}",
            "data": {}
        }

@app.post("/api/generate-exam")
def generate_exam(
    question_count: Optional[int] = Query(10, description="Sınav soru sayısı", ge=5, le=20)
):
    """
    Gerçekçi LGS İngilizce sınavı üret
    
    - **question_count**: Sınav soru sayısı (5-20 arası)
    - Konu dağılımı geçmiş yıllardaki gerçek oranlarla aynı
    - İstatistik tablosundan hızlı hesaplama
    """
    try:
        exam_result = gemini_service.generate_exam_questions(question_count)
        
        if not exam_result.get('success', False):
            return {
                "success": False,
                "message": "Sınav üretilemedi",
                "data": {}
            }
        
        return {
            "success": True,
            "message": f"{question_count} soruluk LGS İngilizce sınavı üretildi",
            "data": {
                "exam_info": exam_result['exam_metadata'],
                "questions": exam_result['questions'],
                "total_questions": len(exam_result['questions'])
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Sınav üretme hatası: {str(e)}",
            "data": {}
        }

@app.get("/api/statistics")
def get_statistics_data(
    year: Optional[int] = Query(None, description="Belirli yıl filtresi"),
    topic: Optional[str] = Query(None, description="Belirli konu filtresi"),
    limit: Optional[int] = Query(50, description="Sonuç sayısı sınırı")
):
    """
    LGS İstatistik tablosundaki verileri getir
    
    - **year**: Belirli yıl (opsiyonel)
    - **topic**: Belirli konu (opsiyonel)
    - **limit**: Maksimum sonuç sayısı (varsayılan: 50)
    """
    try:
        # Database bağlantısını kontrol et
        if not db.connection:
            db.connect()
        
        # Base query
        query = """
            SELECT topic, year, question_count, percentage, 
                   total_questions_in_year, created_at, updated_at
            FROM lgs_statistics 
            WHERE 1=1
        """
        params = []
        
        # Filters
        if year:
            query += " AND year = %s"
            params.append(year)
        
        if topic:
            query += " AND topic ILIKE %s"
            params.append(f"%{topic}%")
        
        # Order and limit
        query += " ORDER BY year DESC, question_count DESC LIMIT %s"
        params.append(limit)
        
        # Execute
        statistics = db.execute_query(query, params)
        
        if statistics is None:
            return {
                "success": False,
                "message": "İstatistik tablosu sorgulanamadı - lütfen create_statistics_table.py çalıştırın",
                "data": [],
                "count": 0
            }
        
        if not statistics:
            return {
                "success": True,
                "message": "İstatistik bulunamadı",
                "data": [],
                "count": 0
            }
        
        stats_list = [dict(stat) for stat in statistics]
        
        return {
            "success": True,
            "message": f"{len(stats_list)} istatistik kaydı bulundu",
            "data": stats_list,
            "count": len(stats_list)
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"İstatistik getirme hatası: {str(e)}",
            "data": [],
            "count": 0
        }

@app.get("/api/statistics/summary")
def get_statistics_summary():
    """
    LGS İstatistik özeti - Konu bazında toplam veriler
    """
    try:
        # Database bağlantısını kontrol et
        if not db.connection:
            db.connect()
        
        # Konu bazında özet
        topic_summary_query = """
            SELECT topic, 
                   COUNT(DISTINCT year) as years_appeared,
                   SUM(question_count) as total_questions,
                   ROUND(AVG(percentage), 2) as avg_percentage,
                   MIN(year) as first_year,
                   MAX(year) as last_year
            FROM lgs_statistics
            GROUP BY topic
            ORDER BY total_questions DESC
        """
        
        topic_summary = db.execute_query(topic_summary_query)
        
        # Yıl bazında özet
        year_summary_query = """
            SELECT year, 
                   COUNT(DISTINCT topic) as topic_count,
                   SUM(question_count) as total_questions,
                   MAX(total_questions_in_year) as total_questions_in_year
            FROM lgs_statistics
            GROUP BY year
            ORDER BY year DESC
        """
        
        year_summary = db.execute_query(year_summary_query)
        
        # Genel istatistikler
        general_stats_query = """
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT topic) as unique_topics,
                COUNT(DISTINCT year) as unique_years,
                SUM(question_count) as total_questions,
                MIN(year) as earliest_year,
                MAX(year) as latest_year
            FROM lgs_statistics
        """
        
        general_stats = db.execute_single_query(general_stats_query)
        
        if topic_summary is None or year_summary is None or general_stats is None:
            return {
                "success": False,
                "message": "İstatistik tablosu bulunamadı - lütfen create_statistics_table.py çalıştırın",
                "data": {}
            }
        
        return {
            "success": True,
            "message": "İstatistik özeti başarıyla getirildi",
            "data": {
                "general_statistics": dict(general_stats) if general_stats else {},
                "topic_summary": [dict(row) for row in topic_summary] if topic_summary else [],
                "year_summary": [dict(row) for row in year_summary] if year_summary else []
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"İstatistik özeti hatası: {str(e)}",
            "data": {}
        }

@app.get("/api/statistics/distribution")
def get_topic_distribution_data():
    """
    Konu dağılım verilerini getir - Sınav üretme için kullanılan veriler
    """
    try:
        # Database bağlantısını kontrol et
        if not db.connection:
            db.connect()
        
        # Son 5 yılın konu dağılımı
        distribution_query = """
            SELECT topic, 
                   ROUND(AVG(percentage), 1) as avg_percentage,
                   SUM(question_count) as total_questions,
                   COUNT(DISTINCT year) as years_appeared,
                   ARRAY_AGG(DISTINCT year ORDER BY year DESC) as years
            FROM lgs_statistics 
            WHERE year >= 2020
            GROUP BY topic 
            HAVING SUM(question_count) >= 3
            ORDER BY total_questions DESC
        """
        
        distribution_data = db.execute_query(distribution_query)
        
        if distribution_data is None:
            return {
                "success": False,
                "message": "Dağılım verisi bulunamadı - lütfen create_statistics_table.py çalıştırın",
                "data": []
            }
        
        if not distribution_data:
            return {
                "success": True,
                "message": "Dağılım verisi bulunamadı",
                "data": []
            }
        
        distribution_list = [dict(row) for row in distribution_data]
        
        # Toplam yüzde hesapla
        total_percentage = sum([float(row['avg_percentage']) for row in distribution_list])
        
        return {
            "success": True,
            "message": f"{len(distribution_list)} konu dağılımı bulundu",
            "data": {
                "topic_distribution": distribution_list,
                "total_percentage": round(total_percentage, 1),
                "based_on_years": "2020+",
                "note": "Bu veriler sınav üretme algoritmasında kullanılır"
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Dağılım verisi hatası: {str(e)}",
            "data": {}
        }

# ============================================
# ML MODEL ENDPOINTLERİ - YENİ!
# ============================================

@app.post("/api/ml/train")
def train_ml_model(
    topic: Optional[str] = Query(None, description="Belirli konu için eğit (opsiyonel)"),
    limit: Optional[int] = Query(200, description="Eğitim verisi sayısı", ge=50, le=500),
    model_type: Optional[str] = Query("ensemble", description="Model tipi: random_forest, gradient_boosting, veya ensemble")
):
    """
    🤖 ML Modelini Eğit
    
    DB'deki geçmiş LGS sorularını kullanarak modeli eğitir.
    
    - **topic**: Belirli konu için eğit (boş bırakırsan tüm konular)
    - **limit**: Kaç soru ile eğitilecek (50-500 arası)
    - **model_type**: Model tipi
    
    **Model Tipleri:**
    - naive_bayes: Az veri için EN İYİ! (70 soru için önerilen) ⭐
    - random_forest: Hızlı, dengeli (200+ soru için)
    - gradient_boosting: Yüksek accuracy (200+ soru için)
    - ensemble: 3 model birleşimi (500+ soru için)
    
    Eğitim sonrası model accuracy, F1 score ve istatistikler döner.
    """
    try:
        if model_type not in ["random_forest", "gradient_boosting", "ensemble"]:
            return {
                "success": False,
                "message": "model_type 'random_forest', 'gradient_boosting' veya 'ensemble' olmalı",
                "stats": {}
            }
        
        result = ml_service.train_model(topic=topic, limit=limit, model_type=model_type)
        return result
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Model eğitim hatası: {str(e)}",
            "stats": {}
        }

@app.post("/api/ml/generate")
def generate_with_ml_model(
    topic: Optional[str] = Query(None, description="Belirli konu (opsiyonel)"),
    count: Optional[int] = Query(5, description="Soru sayısı", ge=1, le=20),
    difficulty: Optional[str] = Query("orta", description="Zorluk: kolay, orta, zor")
):
    """
    🎯 Eğitilmiş Modelle Soru Üret
    
    Önceden eğitilmiş ML modeli kullanarak yeni sorular üretir.
    **DB'ye kaydetmez**, sadece üretir ve döner.
    
    - **topic**: Belirli konu için soru üret (boş bırakırsan karma)
    - **count**: Kaç soru üretilecek (1-20)
    - **difficulty**: Zorluk seviyesi
    
    ⚠️ Not: Önce /api/ml/train endpoint'ini çağırmalısınız!
    
    Dönen veri:
    - Üretilen sorular
    - Model accuracy
    - Başarı oranı
    - Eğitim istatistikleri
    """
    try:
        if difficulty not in ["kolay", "orta", "zor"]:
            return {
                "success": False,
                "message": "Zorluk 'kolay', 'orta' veya 'zor' olmalı",
                "questions": [],
                "model_info": {}
            }
        
        result = ml_service.generate_questions_with_model(
            topic=topic,
            count=count,
            difficulty=difficulty
        )
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Soru üretme hatası: {str(e)}",
            "questions": [],
            "model_info": {}
        }

@app.get("/api/ml/status")
def get_ml_model_status():
    """
    📊 ML Model Durumu
    
    Modelin eğitim durumunu, accuracy'sini ve istatistiklerini gösterir.
    
    Dönen bilgiler:
    - Model eğitildi mi?
    - Model accuracy (%)
    - Eğitim verisi boyutu
    - Veri kalitesi skoru
    - Son eğitim tarihi
    """
    try:
        status = ml_service.get_model_status()
        
        return {
            "success": True,
            "message": "Model durumu başarıyla getirildi",
            "data": status
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Model durumu getirme hatası: {str(e)}",
            "data": {}
        }

@app.post("/api/ml/train-and-generate")
def train_and_generate_questions(
    topic: Optional[str] = Query(None, description="Belirli konu (opsiyonel)"),
    training_limit: Optional[int] = Query(200, description="Eğitim verisi sayısı", ge=50, le=500),
    question_count: Optional[int] = Query(5, description="Üretilecek soru sayısı", ge=1, le=20),
    difficulty: Optional[str] = Query("orta", description="Zorluk: kolay, orta, zor"),
    model_type: Optional[str] = Query("ensemble", description="Model tipi: random_forest, gradient_boosting, veya ensemble")
):
    """
    🚀 Tek Seferde: Eğit + Üret
    
    Gerçek ML modelini eğitir (Random Forest) ve hemen ardından yeni sorular üretir.
    **DB'ye kaydetmez**, sadece üretir.
    
    - **topic**: Belirli konu (boş bırakırsan tüm konular)
    - **training_limit**: Kaç soru ile eğitilecek (50-500)
    - **question_count**: Kaç soru üretilecek (1-20)
    - **difficulty**: Zorluk seviyesi
    
    Dönen veri:
    - Eğitim istatistikleri (TF-IDF, Random Forest)
    - Model accuracy (gerçek ML accuracy)
    - Üretilen sorular
    - Başarı oranı
    """
    try:
        # 1. Modeli eğit
        training_result = ml_service.train_model(topic=topic, limit=training_limit, model_type=model_type)
        
        if not training_result.get('success', False):
            return {
                "success": False,
                "message": "Model eğitilemedi",
                "training_result": training_result,
                "questions": []
            }
        
        # 2. Soru üret
        generation_result = ml_service.generate_questions_with_model(
            topic=topic,
            count=question_count,
            difficulty=difficulty
        )
        
        if not generation_result.get('success', False):
            return {
                "success": False,
                "message": "Sorular üretilemedi",
                "training_result": training_result,
                "generation_result": generation_result
            }
        
        # 3. Birleştirilmiş sonuç
        return {
            "success": True,
            "message": f"ML Model eğitildi ve {len(generation_result['questions'])} soru üretildi",
            "training_stats": training_result['stats'],
            "model_info": generation_result['model_info'],
            "questions": generation_result['questions'],
            "generation_stats": generation_result.get('generation_stats', {}),
            "summary": {
                "training_data_size": training_result['stats'].get('total_questions', 0),
                "model_type": training_result['stats'].get('model_type', 'Random Forest'),
                "model_accuracy": generation_result['model_info'].get('accuracy', 0),
                "generation_success_rate": generation_result['model_info'].get('generation_success_rate', 0),
                "data_quality": generation_result['model_info'].get('data_quality', 0),
                "generated_count": len(generation_result['questions']),
                "top_features": generation_result['model_info'].get('top_features', [])
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Eğitim ve üretim hatası: {str(e)}",
            "training_result": {},
            "questions": []
        }

@app.post("/api/ml/predict-topic")
def predict_question_topic(
    question_text: str = Query(..., description="Soru metni")
):
    """
    🔮 Soru Metninden Konu Tahmini
    
    Eğitilmiş ML modeli kullanarak soru metninden konu tahmini yapar.
    
    - **question_text**: Tahmin edilecek soru metni
    
    Dönen veri:
    - Tahmin edilen konu
    - Güven skoru (%)
    - Top 3 tahmin
    """
    try:
        prediction = ml_service.predict_topic(question_text)
        
        if 'error' in prediction:
            return {
                "success": False,
                "message": prediction['error'],
                "prediction": {}
            }
        
        return {
            "success": True,
            "message": "Konu tahmini başarılı",
            "prediction": prediction,
            "question_text": question_text
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Tahmin hatası: {str(e)}",
            "prediction": {}
        }

@app.get("/api/ml/debug")
def debug_ml_model():
    """
    🔧 ML Model Debug Bilgileri
    
    Model durumu ve debug bilgilerini gösterir.
    """
    try:
        import sys
        import sklearn
        
        return {
            "success": True,
            "debug_info": {
                "python_version": sys.version,
                "sklearn_version": sklearn.__version__,
                "numpy_version": np.__version__,
                "model_status": ml_service.get_model_status(),
                "model_files_exist": {
                    "classifier": (ml_service.model_dir / "topic_classifier.pkl").exists(),
                    "vectorizer": (ml_service.model_dir / "vectorizer.pkl").exists(),
                    "encoder": (ml_service.model_dir / "label_encoder.pkl").exists()
                }
            }
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Debug hatası: {str(e)}"
        }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)