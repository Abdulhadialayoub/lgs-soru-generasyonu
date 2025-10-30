import os
import json
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

load_dotenv()

# Database connection
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        sslmode=os.getenv("SSL_MODE", "require")
    )

def create_weights_cache_table():
    """Ağırlık cache tablosunu oluştur"""
    connection = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        # Cache tablosunu oluştur
        create_table_query = """
        CREATE TABLE IF NOT EXISTS topic_weights_cache (
            id SERIAL PRIMARY KEY,
            topic VARCHAR(255) NOT NULL,
            weight INTEGER NOT NULL,
            total_questions INTEGER NOT NULL,
            percentage DECIMAL(5,2),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(topic)
        );
        """
        
        cursor.execute(create_table_query)
        
        # Index ekle
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_topic_weights_topic ON topic_weights_cache(topic);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_topic_weights_weight ON topic_weights_cache(weight DESC);")
        
        connection.commit()
        cursor.close()
        
        print("✅ topic_weights_cache tablosu başarıyla oluşturuldu!")
        return True
        
    except Exception as e:
        print(f"❌ Tablo oluşturma hatası: {e}")
        if connection:
            connection.rollback()
        return False
    
    finally:
        if connection:
            connection.close()

def calculate_and_save_weights():
    """İstatistik tablosundan ağırlıkları hesapla ve kaydet"""
    connection = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        
        print("📊 Konu ağırlıkları hesaplanıyor...")
        
        # Önce tabloyu temizle
        cursor.execute("DELETE FROM topic_weights_cache;")
        
        # İstatistik tablosundan verileri al
        stats_query = """
            SELECT topic, 
                   SUM(question_count) as total_questions,
                   ROUND(AVG(percentage), 2) as avg_percentage
            FROM lgs_statistics 
            WHERE year >= 2020
            GROUP BY topic 
            HAVING SUM(question_count) >= 3
            ORDER BY total_questions DESC
        """
        
        cursor.execute(stats_query)
        stats_results = cursor.fetchall()
        
        if not stats_results:
            print("❌ İstatistik tablosunda veri bulunamadı!")
            return False
        
        # Ağırlıkları hesapla
        max_questions = 0
        topic_data = []
        
        # En çok soru sayısını bul
        for stat in stats_results:
            total_questions = stat['total_questions']
            if total_questions > max_questions:
                max_questions = total_questions
            topic_data.append(dict(stat))
        
        # Ağırlıkları hesapla ve kaydet
        insert_query = """
            INSERT INTO topic_weights_cache (topic, weight, total_questions, percentage)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (topic) 
            DO UPDATE SET 
                weight = EXCLUDED.weight,
                total_questions = EXCLUDED.total_questions,
                percentage = EXCLUDED.percentage,
                updated_at = CURRENT_TIMESTAMP;
        """
        
        inserted_count = 0
        weights_summary = {}
        
        for topic_info in topic_data:
            topic = topic_info['topic']
            total_questions = topic_info['total_questions']
            avg_percentage = topic_info['avg_percentage']
            
            # Ağırlığı hesapla (1-25 arası)
            weight = max(1, int((total_questions / max_questions) * 25))
            
            cursor.execute(insert_query, (topic, weight, total_questions, avg_percentage))
            inserted_count += 1
            weights_summary[topic] = weight
        
        connection.commit()
        cursor.close()
        
        print(f"✅ {inserted_count} konu ağırlığı başarıyla kaydedildi!")
        print(f"📈 Hesaplanan ağırlıklar: {weights_summary}")
        
        # Özet bilgileri göster
        show_weights_summary()
        
        return True
        
    except Exception as e:
        print(f"❌ Ağırlık hesaplama hatası: {e}")
        if connection:
            connection.rollback()
        return False
    
    finally:
        if connection:
            connection.close()

def show_weights_summary():
    """Ağırlık özetini göster"""
    connection = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        
        # Ağırlık özetini al
        cursor.execute("""
            SELECT topic, weight, total_questions, percentage, created_at
            FROM topic_weights_cache 
            ORDER BY weight DESC;
        """)
        weights = cursor.fetchall()
        
        print(f"\n🎯 KONU AĞIRLIK ÖZETİ:")
        print("=" * 60)
        print(f"{'Konu':<20} {'Ağırlık':<8} {'Soru Sayısı':<12} {'Yüzde':<8}")
        print("-" * 60)
        
        for weight_info in weights:
            topic = weight_info['topic'][:18]  # Kısalt
            weight = weight_info['weight']
            total_q = weight_info['total_questions']
            percentage = weight_info['percentage']
            
            print(f"{topic:<20} {weight:<8} {total_q:<12} {percentage}%")
        
        print("-" * 60)
        print(f"Toplam konu sayısı: {len(weights)}")
        print(f"Güncelleme tarihi: {weights[0]['created_at'] if weights else 'N/A'}")
        
        cursor.close()
        
    except Exception as e:
        print(f"❌ Özet gösterme hatası: {e}")
    
    finally:
        if connection:
            connection.close()

def main():
    """Ana fonksiyon"""
    print("🚀 LGS Konu Ağırlık Hesaplayıcı")
    print("=" * 50)
    
    # 1. Tabloyu oluştur
    if not create_weights_cache_table():
        print("❌ Tablo oluşturulamadı, işlem durduruluyor.")
        return
    
    # 2. Ağırlıkları hesapla ve kaydet
    if not calculate_and_save_weights():
        print("❌ Ağırlıklar hesaplanamadı, işlem durduruluyor.")
        return
    
    print("\n🎉 İşlem tamamlandı!")
    print("Artık sınav üretme API'si bu ağırlıkları kullanacak!")
    print("\nKullanım:")
    print("- POST /api/generate-exam?question_count=10")

if __name__ == "__main__":
    main()