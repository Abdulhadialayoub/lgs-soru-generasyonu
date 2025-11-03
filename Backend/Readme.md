# LGS Soru Tahmin API - Backend

Bu klasör, LGS İngilizce soru tahmin projesinin FastAPI backend kısmını içerir.

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. `.env` dosyasının doğru yapılandırıldığından emin olun.

3. Uygulamayı çalıştırın:
```bash
python main.py
```

veya

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### Genel Endpoints
- **GET /** - Ana sayfa
- **GET /api/questions** - Tüm soruları getir (filtreleme seçenekleri ile)
- **GET /api/questions/{id}** - Belirli bir soruyu getir
- **GET /api/stats** - Soru istatistikleri

### Soru Üretme (Gemini)
- **POST /api/predict** - Soru dağılımı tahmini
- **POST /api/generate** - Soru üretme (belirli konu)
- **POST /api/generate-exam** - Gerçekçi LGS sınavı üret

### İstatistik Endpoints
- **GET /api/statistics** - İstatistik tablosu verileri
- **GET /api/statistics/summary** - İstatistik özeti
- **GET /api/statistics/distribution** - Konu dağılım verileri

### 🤖 ML Model Endpoints (YENİ!)
- **POST /api/ml/train** - Modeli DB'deki sorularla eğit
- **POST /api/ml/generate** - Eğitilmiş modelle soru üret (DB'ye kaydetmez)
- **GET /api/ml/status** - Model durumu ve accuracy
- **POST /api/ml/train-and-generate** - Tek seferde eğit + üret

## Swagger Dokümantasyonu

Uygulama çalıştıktan sonra şu adreslerde API dokümantasyonuna erişebilirsiniz:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Veritabanı

Proje Supabase PostgreSQL kullanmaktadır. Veritabanı bağlantı bilgileri `.env` dosyasında tanımlanmıştır.


## 🤖 ML Model Kullanımı

### Adım 1: Modeli Eğit
```bash
POST /api/ml/train
```
Parametreler:
- `topic` (opsiyonel): Belirli konu için eğit
- `limit` (varsayılan: 200): Kaç soru ile eğitilecek

Dönen veri:
- Model accuracy (%)
- Eğitim istatistikleri
- Veri kalitesi skoru
- Konu dağılımı

### Adım 2: Soru Üret
```bash
POST /api/ml/generate
```
Parametreler:
- `topic` (opsiyonel): Belirli konu
- `count` (varsayılan: 5): Kaç soru üretilecek
- `difficulty` (varsayılan: orta): kolay/orta/zor

Dönen veri:
- Üretilen sorular (DB'ye kaydetmez!)
- Model accuracy
- Başarı oranı
- Eğitim istatistikleri

### Tek Seferde Eğit + Üret
```bash
POST /api/ml/train-and-generate
```
Parametreler:
- `topic` (opsiyonel): Belirli konu
- `training_limit` (varsayılan: 200): Eğitim verisi
- `question_count` (varsayılan: 5): Üretilecek soru sayısı
- `difficulty` (varsayılan: orta): Zorluk

### Model Durumu Kontrol
```bash
GET /api/ml/status
```

Dönen veri:
- Model eğitildi mi?
- Model accuracy
- Eğitim verisi boyutu
- Son eğitim tarihi

## Özellikler

### ML Model Özellikleri
✅ DB'deki geçmiş sorulardan öğrenir
✅ Model accuracy hesaplar (%75-100 arası)
✅ Veri kalitesi analizi yapar
✅ Başarı oranı gösterir
✅ DB'ye kaydetmeden soru üretir
✅ Konu bazlı veya karma eğitim
✅ Few-shot learning ile kaliteli sorular

### Başarı Metrikleri
- **Model Accuracy**: Eğitim verisinin kalitesi ve çeşitliliğine göre
- **Data Quality Score**: Eksik alan, konu çeşitliliği kontrolü
- **Generation Success Rate**: Üretilen soruların geçerlilik oranı

## Örnek Kullanım

### Python ile
```python
import requests

# 1. Modeli eğit
train_response = requests.post(
    "http://localhost:8000/api/ml/train",
    params={"topic": "Teen Life", "limit": 150}
)
print(train_response.json())

# 2. Soru üret
generate_response = requests.post(
    "http://localhost:8000/api/ml/generate",
    params={"topic": "Teen Life", "count": 5, "difficulty": "orta"}
)
print(generate_response.json())
```

### cURL ile
```bash
# Tek seferde eğit + üret
curl -X POST "http://localhost:8000/api/ml/train-and-generate?topic=Friendship&training_limit=200&question_count=10&difficulty=orta"
```
