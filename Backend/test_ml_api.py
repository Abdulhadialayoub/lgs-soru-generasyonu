"""
ML API Test Script
Yeni ML endpoint'lerini test eder
"""
import requests
import json
from time import sleep

BASE_URL = "http://localhost:8000"

def print_section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def test_ml_status():
    """Model durumunu kontrol et"""
    print_section("1. Model Durumu Kontrolü")
    
    response = requests.get(f"{BASE_URL}/api/ml/status")
    data = response.json()
    
    print(f"✅ Status Code: {response.status_code}")
    print(f"📊 Model Eğitildi mi: {data['data']['is_trained']}")
    print(f"📊 Model Accuracy: {data['data']['model_accuracy']}%")
    print(f"📊 Eğitim Verisi: {data['data']['training_data_size']} soru")
    
    return data

def test_ml_train(topic=None, limit=100):
    """Gerçek ML modelini eğit"""
    print_section(f"2. ML Model Eğitimi (Konu: {topic or 'Tümü'}, Limit: {limit})")
    
    params = {"limit": limit}
    if topic:
        params["topic"] = topic
    
    response = requests.post(f"{BASE_URL}/api/ml/train", params=params)
    data = response.json()
    
    print(f"✅ Status Code: {response.status_code}")
    print(f"✅ Başarılı: {data['success']}")
    print(f"📊 Mesaj: {data['message']}")
    
    if data['success']:
        stats = data['stats']
        print(f"\n📈 ML Model İstatistikleri:")
        print(f"  - Model Tipi: {stats.get('model_type', 'N/A')}")
        print(f"  - Toplam Soru: {stats['total_questions']}")
        print(f"  - Konu Çeşitliliği: {stats['unique_topics']} farklı konu")
        print(f"  - Topic Accuracy: {stats.get('topic_accuracy', 0)}%")
        print(f"  - Train Size: {stats.get('train_size', 0)}")
        print(f"  - Test Size: {stats.get('test_size', 0)}")
        print(f"  - Veri Kalitesi: {stats['data_quality_score']}%")
        
        print(f"\n🔑 Top Features (TF-IDF):")
        for feature in stats.get('top_features', [])[:5]:
            print(f"  - {feature}")
        
        print(f"\n📚 Konu Dağılımı:")
        for topic, count in list(stats['topic_distribution'].items())[:5]:
            print(f"  - {topic}: {count} soru")
    
    return data

def test_ml_generate(topic=None, count=5, difficulty="orta"):
    """Eğitilmiş modelle soru üret"""
    print_section(f"3. Soru Üretimi (Konu: {topic or 'Karma'}, Adet: {count})")
    
    params = {
        "count": count,
        "difficulty": difficulty
    }
    if topic:
        params["topic"] = topic
    
    response = requests.post(f"{BASE_URL}/api/ml/generate", params=params)
    data = response.json()
    
    print(f"✅ Status Code: {response.status_code}")
    print(f"✅ Başarılı: {data['success']}")
    print(f"📊 Mesaj: {data['message']}")
    
    if data['success']:
        model_info = data['model_info']
        print(f"\n🤖 Model Bilgileri:")
        print(f"  - Model Accuracy: {model_info['accuracy']}%")
        print(f"  - Eğitim Verisi: {model_info['training_data_size']} soru")
        print(f"  - Üretim Başarı Oranı: {model_info['generation_success_rate']}%")
        print(f"  - Veri Kalitesi: {model_info['data_quality']}%")
        
        print(f"\n📝 Üretilen Sorular ({len(data['questions'])} adet):")
        for i, q in enumerate(data['questions'][:2], 1):  # İlk 2 soruyu göster
            print(f"\n  Soru {i}:")
            print(f"  Konu: {q.get('topic', 'N/A')}")
            print(f"  Soru: {q.get('question_text', 'N/A')[:80]}...")
            print(f"  A) {q.get('option_a', 'N/A')[:40]}...")
            print(f"  B) {q.get('option_b', 'N/A')[:40]}...")
            print(f"  C) {q.get('option_c', 'N/A')[:40]}...")
            print(f"  D) {q.get('option_d', 'N/A')[:40]}...")
            print(f"  Doğru: {q.get('correct_option', 'N/A')}")
            print(f"  Açıklama: {q.get('explanation', 'N/A')[:60]}...")
    
    return data

def test_train_and_generate(topic=None, training_limit=150, question_count=5):
    """Tek seferde eğit + üret"""
    print_section(f"4. Tek Seferde Eğit + Üret (Konu: {topic or 'Tümü'})")
    
    params = {
        "training_limit": training_limit,
        "question_count": question_count,
        "difficulty": "orta"
    }
    if topic:
        params["topic"] = topic
    
    response = requests.post(f"{BASE_URL}/api/ml/train-and-generate", params=params)
    data = response.json()
    
    print(f"✅ Status Code: {response.status_code}")
    print(f"✅ Başarılı: {data['success']}")
    print(f"📊 Mesaj: {data['message']}")
    
    if data['success']:
        summary = data['summary']
        print(f"\n📊 Özet:")
        print(f"  - Eğitim Verisi: {summary['training_data_size']} soru")
        print(f"  - Model Accuracy: {summary['model_accuracy']}%")
        print(f"  - Üretim Başarı Oranı: {summary['generation_success_rate']}%")
        print(f"  - Veri Kalitesi: {summary['data_quality']}%")
        print(f"  - Üretilen Soru: {summary['generated_count']} adet")
        
        print(f"\n📝 Örnek Soru:")
        if data['questions']:
            q = data['questions'][0]
            print(f"  Konu: {q.get('topic', 'N/A')}")
            print(f"  Soru: {q.get('question_text', 'N/A')}")
            print(f"  A) {q.get('option_a', 'N/A')}")
            print(f"  B) {q.get('option_b', 'N/A')}")
            print(f"  C) {q.get('option_c', 'N/A')}")
            print(f"  D) {q.get('option_d', 'N/A')}")
            print(f"  Doğru: {q.get('correct_option', 'N/A')}")
    
    return data

def test_predict_topic():
    """Soru metninden konu tahmini"""
    print_section("5. Konu Tahmini (ML Model)")
    
    test_questions = [
        "What do you usually do in your free time?",
        "My best friend always helps me with my homework.",
        "I use the internet to search for information."
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n🔮 Test {i}: {question}")
        
        response = requests.post(
            f"{BASE_URL}/api/ml/predict-topic",
            params={"question_text": question}
        )
        data = response.json()
        
        if data['success']:
            pred = data['prediction']
            print(f"  ✅ Tahmin: {pred['predicted_topic']}")
            print(f"  📊 Güven: {pred['confidence']}%")
            print(f"  🏆 Top 3:")
            for top in pred['top_3_predictions']:
                print(f"    - {top['topic']}: {top['confidence']:.1f}%")

def main():
    """Ana test fonksiyonu"""
    print("\n🚀 ML API Test Başlıyor...")
    print(f"🌐 Base URL: {BASE_URL}")
    print("🤖 Gerçek ML Model (Random Forest + TF-IDF)")
    
    try:
        # 1. Model durumu
        test_ml_status()
        sleep(1)
        
        # 2. Model eğitimi (Gerçek ML!)
        test_ml_train(topic="Teen Life", limit=100)
        sleep(2)
        
        # 3. Konu tahmini
        test_predict_topic()
        sleep(1)
        
        # 4. Soru üretimi
        test_ml_generate(topic="Teen Life", count=3)
        sleep(2)
        
        # 5. Tek seferde eğit + üret
        test_train_and_generate(topic="Friendship", training_limit=150, question_count=5)
        
        print_section("✅ Tüm Testler Tamamlandı!")
        print("\n💾 Model kaydedildi: ml_models/ klasöründe")
        print("🔄 API yeniden başlatıldığında model otomatik yüklenecek")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ HATA: API'ye bağlanılamadı!")
        print("Lütfen önce API'yi başlatın: python main.py")
    except Exception as e:
        print(f"\n❌ HATA: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
