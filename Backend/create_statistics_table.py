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

def create_statistics_table():
    """İstatistik tablosunu oluştur"""
    connection = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        # İstatistik tablosunu oluştur
        create_table_query = """
        CREATE TABLE IF NOT EXISTS lgs_statistics (
            id SERIAL PRIMARY KEY,
            topic VARCHAR(255) NOT NULL,
            year INTEGER NOT NULL,
            question_count INTEGER NOT NULL,
            percentage DECIMAL(5,2),
            total_questions_in_year INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(topic, year)
        );
        """
        
        cursor.execute(create_table_query)
        
        # Index'ler ekle (hızlandırma için)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_lgs_statistics_topic ON lgs_statistics(topic);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_lgs_statistics_year ON lgs_statistics(year);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_lgs_statistics_topic_year ON lgs_statistics(topic, year);")
        
        connection.commit()
        cursor.close()
        
        print("✅ lgs_statistics tablosu başarıyla oluşturuldu!")
        return True
        
    except Exception as e:
        print(f"❌ Tablo oluşturma hatası: {e}")
        if connection:
            connection.rollback()
        return False
    
    finally:
        if connection:
            connection.close()

def populate_statistics_table():
    """İstatistik tablosunu verilerle doldur"""
    connection = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        
        print("📊 İstatistikler hesaplanıyor...")
        
        # Önce tabloyu temizle
        cursor.execute("DELETE FROM lgs_statistics;")
        
        # Yıl ve konu bazında istatistikleri hesapla
        stats_query = """
        WITH year_totals AS (
            SELECT year, COUNT(*) as total_in_year
            FROM lgs_questions 
            GROUP BY year
        ),
        topic_year_counts AS (
            SELECT 
                topic,
                year,
                COUNT(*) as question_count
            FROM lgs_questions 
            GROUP BY topic, year
        )
        SELECT 
            tyc.topic,
            tyc.year,
            tyc.question_count,
            ROUND((tyc.question_count * 100.0 / yt.total_in_year), 2) as percentage,
            yt.total_in_year as total_questions_in_year
        FROM topic_year_counts tyc
        JOIN year_totals yt ON tyc.year = yt.year
        ORDER BY tyc.year DESC, tyc.question_count DESC;
        """
        
        cursor.execute(stats_query)
        statistics = cursor.fetchall()
        
        if not statistics:
            print("❌ İstatistik verisi bulunamadı!")
            return False
        
        # İstatistikleri tabloya ekle
        insert_query = """
        INSERT INTO lgs_statistics (topic, year, question_count, percentage, total_questions_in_year)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (topic, year) 
        DO UPDATE SET 
            question_count = EXCLUDED.question_count,
            percentage = EXCLUDED.percentage,
            total_questions_in_year = EXCLUDED.total_questions_in_year,
            updated_at = CURRENT_TIMESTAMP;
        """
        
        inserted_count = 0
        for stat in statistics:
            cursor.execute(insert_query, (
                stat['topic'],
                stat['year'],
                stat['question_count'],
                stat['percentage'],
                stat['total_questions_in_year']
            ))
            inserted_count += 1
        
        connection.commit()
        cursor.close()
        
        print(f"✅ {inserted_count} istatistik kaydı başarıyla eklendi!")
        
        # Özet bilgileri göster
        show_summary()
        
        return True
        
    except Exception as e:
        print(f"❌ Veri doldurma hatası: {e}")
        if connection:
            connection.rollback()
        return False
    
    finally:
        if connection:
            connection.close()

def show_summary():
    """İstatistik özetini göster"""
    connection = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor(cursor_factory=RealDictCursor)
        
        # Toplam istatistikler
        cursor.execute("SELECT COUNT(*) as total_records FROM lgs_statistics;")
        total_records = cursor.fetchone()['total_records']
        
        # Yıl bazında özet
        cursor.execute("""
            SELECT year, COUNT(DISTINCT topic) as topic_count, SUM(question_count) as total_questions
            FROM lgs_statistics 
            GROUP BY year 
            ORDER BY year DESC 
            LIMIT 5;
        """)
        year_summary = cursor.fetchall()
        
        # Konu bazında özet
        cursor.execute("""
            SELECT topic, COUNT(DISTINCT year) as year_count, SUM(question_count) as total_questions
            FROM lgs_statistics 
            GROUP BY topic 
            ORDER BY total_questions DESC 
            LIMIT 10;
        """)
        topic_summary = cursor.fetchall()
        
        print(f"\n📈 İSTATİSTİK ÖZETİ:")
        print(f"Toplam kayıt sayısı: {total_records}")
        
        print(f"\n📅 YIL BAZINDA ÖZET:")
        for year_stat in year_summary:
            print(f"  {year_stat['year']}: {year_stat['topic_count']} konu, {year_stat['total_questions']} soru")
        
        print(f"\n📚 KONU BAZINDA ÖZET (İlk 10):")
        for topic_stat in topic_summary:
            print(f"  {topic_stat['topic']}: {topic_stat['year_count']} yıl, {topic_stat['total_questions']} soru")
        
        cursor.close()
        
    except Exception as e:
        print(f"❌ Özet gösterme hatası: {e}")
    
    finally:
        if connection:
            connection.close()

def create_summary_view():
    """Hızlı erişim için view oluştur"""
    connection = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        # View oluştur
        view_query = """
        CREATE OR REPLACE VIEW lgs_topic_summary AS
        SELECT 
            topic,
            COUNT(DISTINCT year) as years_appeared,
            SUM(question_count) as total_questions,
            ROUND(AVG(percentage), 2) as avg_percentage,
            MIN(year) as first_year,
            MAX(year) as last_year,
            ROUND(SUM(question_count) * 100.0 / (SELECT SUM(question_count) FROM lgs_statistics), 2) as overall_percentage
        FROM lgs_statistics
        GROUP BY topic
        ORDER BY total_questions DESC;
        """
        
        cursor.execute(view_query)
        connection.commit()
        cursor.close()
        
        print("✅ lgs_topic_summary view'i oluşturuldu!")
        
    except Exception as e:
        print(f"❌ View oluşturma hatası: {e}")
        if connection:
            connection.rollback()
    
    finally:
        if connection:
            connection.close()

def main():
    """Ana fonksiyon"""
    print("🚀 LGS İstatistik Tablosu Oluşturucu")
    print("=" * 50)
    
    # 1. Tabloyu oluştur
    if not create_statistics_table():
        print("❌ Tablo oluşturulamadı, işlem durduruluyor.")
        return
    
    # 2. Verileri doldur
    if not populate_statistics_table():
        print("❌ Veriler doldurulamadı, işlem durduruluyor.")
        return
    
    # 3. View oluştur
    create_summary_view()
    
    print("\n🎉 İşlem tamamlandı!")
    print("Artık API'ler çok daha hızlı çalışacak!")
    print("\nKullanılabilir tablolar:")
    print("- lgs_statistics: Detaylı istatistikler")
    print("- lgs_topic_summary: Konu özetleri (view)")

if __name__ == "__main__":
    main()