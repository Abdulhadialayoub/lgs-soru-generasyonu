import os
import time
import json
from dotenv import load_dotenv
import google.generativeai as genai
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any

# Environment variables yükle
load_dotenv()

# Environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
SSL_MODE = os.getenv("SSL_MODE", "require")

def setup_connections():
    """PostgreSQL ve Google AI bağlantılarını kur"""
    
    # Google AI yapılandırması
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable bulunamadı!")
    
    genai.configure(api_key=GOOGLE_API_KEY)
    
    # PostgreSQL bağlantısı
    try:
        connection = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            sslmode=SSL_MODE
        )
        return connection
    except Exception as e:
        raise Exception(f"PostgreSQL bağlantısı kurulamadı: {e}")

def get_questions_without_embeddings(connection) -> List[Dict[str, Any]]:
    """Embedding'i olmayan soruları getir"""
    try:
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        
        # Embedding sütunu NULL olan soruları getir
        query = "SELECT id, question_text FROM lgs_questions WHERE embedding IS NULL"
        cursor.execute(query)
        
        questions = cursor.fetchall()
        cursor.close()
        
        if questions:
            print(f"Embedding'i olmayan {len(questions)} soru bulundu.")
            return [dict(q) for q in questions]
        else:
            print("Embedding'i olmayan soru bulunamadı.")
            return []
            
    except Exception as e:
        print(f"Sorular getirilirken hata oluştu: {e}")
        return []

def create_embeddings_batch(questions: List[str]) -> List[List[float]]:
    """Soru listesi için embedding'ler oluştur"""
    try:
        embeddings = []
        
        for i, question in enumerate(questions):
            print(f"  📝 Soru {i+1}/{len(questions)} işleniyor...")
            
            # Her soru için embedding oluştur
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=question,
                task_type="RETRIEVAL_DOCUMENT"
            )
            embeddings.append(result['embedding'])
            
            # API rate limit için kısa bekleme
            time.sleep(0.1)
        
        return embeddings
        
    except Exception as e:
        print(f"Embedding oluşturulurken hata: {e}")
        return []

def update_embeddings_in_db(connection, question_ids: List[int], embeddings: List[List[float]]):
    """Veritabanındaki embedding'leri güncelle"""
    try:
        cursor = connection.cursor()
        updated_count = 0
        
        for question_id, embedding in zip(question_ids, embeddings):
            # Embedding'i JSON string olarak kaydet
            embedding_json = json.dumps(embedding)
            
            # PostgreSQL'de vector tipine çevir
            query = """
                UPDATE lgs_questions 
                SET embedding = %s::vector 
                WHERE id = %s
            """
            
            cursor.execute(query, (embedding_json, question_id))
            
            if cursor.rowcount > 0:
                updated_count += 1
        
        # Değişiklikleri kaydet
        connection.commit()
        cursor.close()
        
        return updated_count
        
    except Exception as e:
        print(f"Veritabanı güncellenirken hata: {e}")
        connection.rollback()
        return 0

def process_embeddings():
    """Ana işlem fonksiyonu"""
    print("🚀 Embedding oluşturma işlemi başlatılıyor...")
    
    connection = None
    
    try:
        # Bağlantıları kur
        connection = setup_connections()
        print("✅ PostgreSQL ve Google AI bağlantıları kuruldu.")
        
        # Embedding'i olmayan soruları getir
        questions_data = get_questions_without_embeddings(connection)
        
        if not questions_data:
            print("✅ Tüm sorular zaten embedding'e sahip!")
            return
        
        total_questions = len(questions_data)
        batch_size = 10  # API limitleri için daha küçük batch
        total_updated = 0
        
        print(f"📊 Toplam {total_questions} soru işlenecek, {batch_size}'lük gruplar halinde...")
        
        # Küçük gruplar halinde işle
        for i in range(0, total_questions, batch_size):
            batch_end = min(i + batch_size, total_questions)
            batch_data = questions_data[i:batch_end]
            
            print(f"🔄 Grup {i//batch_size + 1} işleniyor... ({i+1}-{batch_end} arası)")
            
            # Bu gruptaki soruların metinlerini ve ID'lerini al
            batch_questions = [item['question_text'] for item in batch_data]
            batch_ids = [item['id'] for item in batch_data]
            
            # Embedding'leri oluştur
            embeddings = create_embeddings_batch(batch_questions)
            
            if embeddings and len(embeddings) == len(batch_questions):
                # Veritabanını güncelle
                updated_count = update_embeddings_in_db(connection, batch_ids, embeddings)
                total_updated += updated_count
                
                print(f"✅ Grup {i//batch_size + 1} tamamlandı. {updated_count} soru güncellendi.")
            else:
                print(f"❌ Grup {i//batch_size + 1} için embedding oluşturulamadı!")
            
            # Gruplar arası bekleme (API rate limit için)
            if batch_end < total_questions:
                print("⏳ Sonraki grup için 5 saniye bekleniyor...")
                time.sleep(5)
        
        print(f"🎉 İşlem tamamlandı! Toplam {total_updated} soru güncellendi.")
        
    except Exception as e:
        print(f"❌ Genel hata oluştu: {e}")
    
    finally:
        # Bağlantıyı kapat
        if connection:
            connection.close()
            print("🔌 Veritabanı bağlantısı kapatıldı.")

if __name__ == "__main__":
    process_embeddings()