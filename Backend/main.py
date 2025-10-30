from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from database import db
from models import Question, QuestionResponse
from typing import Optional
import uvicorn

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

@app.get("/", tags=["Ana Sayfa"])
async def read_root():
    """Ana sayfa endpoint'i"""
    return {
        "message": "LGS Soru Tahmin API'sine Hoş Geldiniz!",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/api/questions", response_model=QuestionResponse, tags=["Sorular"])
async def get_all_questions(
    year: Optional[int] = Query(None, description="Belirli bir yıla göre filtrele"),
    topic: Optional[str] = Query(None, description="Belirli bir konuya göre filtrele"),
    limit: Optional[int] = Query(None, description="Sonuç sayısını sınırla"),
    offset: Optional[int] = Query(0, description="Kaç sonuç atlanacak")
):
    """
    Tüm çıkmış soruları getir
    
    - **year**: Belirli bir yıla göre filtrele (opsiyonel)
    - **topic**: Belirli bir konuya göre filtrele (opsiyonel)  
    - **limit**: Sonuç sayısını sınırla (opsiyonel)
    - **offset**: Kaç sonuç atlanacak (varsayılan: 0)
    """
    try:
        # Base query
        query = "SELECT * FROM lgs_questions WHERE 1=1"
        params = []
        count_query = "SELECT COUNT(*) FROM lgs_questions WHERE 1=1"
        count_params = []
        
        # Year filter
        if year:
            query += " AND year = %s"
            count_query += " AND year = %s"
            params.append(year)
            count_params.append(year)
        
        # Topic filter
        if topic:
            query += " AND topic ILIKE %s"
            count_query += " AND topic ILIKE %s"
            params.append(f"%{topic}%")
            count_params.append(f"%{topic}%")
        
        # Order by
        query += " ORDER BY year DESC, question_number ASC"
        
        # Limit and offset
        if limit:
            query += " LIMIT %s"
            params.append(limit)
        
        if offset:
            query += " OFFSET %s"
            params.append(offset)
        
        # Execute queries
        questions = db.execute_query(query, params)
        total_count_result = db.execute_single_query(count_query, count_params)
        
        if questions is None:
            raise HTTPException(status_code=500, detail="Veritabanı sorgusu başarısız")
        
        total_count = total_count_result['count'] if total_count_result else 0
        
        # Convert to list of dicts
        questions_list = [dict(question) for question in questions]
        
        return QuestionResponse(
            success=True,
            message=f"{len(questions_list)} soru başarıyla getirildi",
            data=questions_list,
            total_count=total_count
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sunucu hatası: {str(e)}")

@app.get("/api/questions/{question_id}", response_model=QuestionResponse, tags=["Sorular"])
async def get_question_by_id(question_id: int):
    """
    ID'ye göre belirli bir soruyu getir
    
    - **question_id**: Getirilecek sorunun ID'si
    """
    try:
        query = "SELECT * FROM lgs_questions WHERE id = %s"
        question = db.execute_single_query(query, (question_id,))
        
        if not question:
            raise HTTPException(status_code=404, detail="Soru bulunamadı")
        
        return QuestionResponse(
            success=True,
            message="Soru başarıyla getirildi",
            data=[dict(question)],
            total_count=1
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sunucu hatası: {str(e)}")

@app.get("/api/stats", tags=["İstatistikler"])
async def get_statistics():
    """
    Veritabanındaki soru istatistiklerini getir
    """
    try:
        # Total questions
        total_query = "SELECT COUNT(*) as total FROM lgs_questions"
        total_result = db.execute_single_query(total_query)
        
        # Questions by year
        year_query = "SELECT year, COUNT(*) as count FROM lgs_questions GROUP BY year ORDER BY year DESC"
        year_results = db.execute_query(year_query)
        
        # Questions by topic
        topic_query = "SELECT topic, COUNT(*) as count FROM lgs_questions GROUP BY topic ORDER BY count DESC"
        topic_results = db.execute_query(topic_query)
        
        return {
            "success": True,
            "message": "İstatistikler başarıyla getirildi",
            "data": {
                "total_questions": total_result['total'] if total_result else 0,
                "questions_by_year": [dict(row) for row in year_results] if year_results else [],
                "questions_by_topic": [dict(row) for row in topic_results] if topic_results else []
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sunucu hatası: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)