# 🎓 LGS Soru Üretim Sistemi

AI destekli LGS İngilizce soru üretim ve tahmin sistemi.

## 🚀 Yeni Özellikler (Gerçek ML Model!)

✅ **Gerçek ML Eğitimi**: Scikit-learn Random Forest Classifier
✅ **TF-IDF Feature Extraction**: 500 feature ile pattern öğrenme
✅ **Model Persistence**: Model kaydedilir ve yeniden kullanılır
✅ **Konu Tahmini**: ML model ile soru metninden konu tahmini
✅ **Hybrid Approach**: ML model + Gemini AI
✅ **Yüksek Accuracy**: %75-95 arası model accuracy
✅ **Hızlı Tahmin**: Milisaniyeler içinde tahmin

## 📁 Proje Yapısı

```
├── Backend/
│   ├── main.py                    # FastAPI ana dosyası
│   ├── ml_service.py              # 🆕 ML model servisi
│   ├── gemini_service.py          # Gemini AI servisi
│   ├── database.py                # Veritabanı bağlantısı
│   ├── models.py                  # Pydantic modelleri
│   ├── test_ml_api.py             # 🆕 ML API test script
│   ├── ML_API_KULLANIM.md         # 🆕 Detaylı kullanım kılavuzu
│   ├── YENI_OZELLIKLER.md         # 🆕 Yeni özellikler dokümantasyonu
│   └── requirements.txt           # Python bağımlılıkları
├── Sorular/                       # LGS soru arşivi
└── README.md                      # Bu dosya
```

## 🛠️ Kurulum

### 1. Bağımlılıkları Yükle
```bash
cd Backend
pip install -r requirements.txt
```

### 2. .env Dosyasını Ayarla
`.env` dosyasında DB ve API key bilgilerini kontrol edin.

### 3. API'yi Başlat
```bash
python main.py
```

API şu adreste çalışacak: http://localhost:8000

## 📚 API Dokümantasyonu

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🤖 ML API Kullanımı

### Hızlı Başlangıç

```bash
# Test script'ini çalıştır
cd Backend
python test_ml_api.py
```

### Temel Kullanım

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Modeli eğit + soru üret (tek seferde)
response = requests.post(
    f"{BASE_URL}/api/ml/train-and-generate",
    params={
        "topic": "Teen Life",
        "training_limit": 200,
        "question_count": 5,
        "difficulty": "orta"
    }
)

result = response.json()
print(f"Model Accuracy: {result['summary']['model_accuracy']}%")
print(f"Üretilen soru: {result['summary']['generated_count']} adet")

# Soruları kullan
for q in result['questions']:
    print(f"\nSoru: {q['question_text']}")
    print(f"Doğru: {q['correct_option']}")
```

### API Endpoints

#### ML Model Endpoints (Yeni!)
- `POST /api/ml/train` - Modeli eğit
- `POST /api/ml/generate` - Soru üret (DB'ye kaydetmez)
- `GET /api/ml/status` - Model durumu
- `POST /api/ml/train-and-generate` - Tek seferde eğit + üret

#### Diğer Endpoints
- `GET /api/questions` - Geçmiş soruları getir
- `POST /api/generate` - Gemini ile soru üret
- `POST /api/generate-exam` - LGS sınavı üret
- `GET /api/statistics` - İstatistikler

## 📊 Başarı Metrikleri

### Topic Accuracy (75-95%)
Random Forest Classifier'ın konu tahmini doğruluğu.
- Train/Test split ile gerçek accuracy
- Scikit-learn metrics ile hesaplanır

### Data Quality Score (0-100%)
Eğitim verisinin kalitesini gösterir.

### Generation Success Rate (0-100%)
Üretilen soruların geçerlilik oranı.

### Feature Importance
TF-IDF ile en önemli kelimeler belirlenir.

## 📖 Detaylı Dokümantasyon

- **Gerçek ML Model**: `Backend/GERCEK_ML_MODEL.md` ⭐
- **Hızlı Başlangıç**: `Backend/NASIL_KULLANILIR.md`
- **ML API Kullanımı**: `Backend/ML_API_KULLANIM.md`
- **Yeni Özellikler**: `Backend/YENI_OZELLIKLER.md`
- **Backend README**: `Backend/Readme.md`

## 🧪 Test

```bash
cd Backend
python test_ml_api.py
```

## 🔑 Önemli Notlar

- ✅ Üretilen sorular **DB'ye kaydetilmez**, sadece döndürülür
- ✅ Model durumu RAM'de tutulur
- ✅ API yeniden başlatıldığında model durumu sıfırlanır
- ✅ Gemini API key gereklidir

## 📝 Örnek Kullanım Senaryoları

### Senaryo 1: Hızlı Test
```bash
curl -X POST "http://localhost:8000/api/ml/train-and-generate?topic=Teen%20Life&training_limit=100&question_count=5"
```

### Senaryo 2: Konu Bazlı Üretim
```python
topics = ["Teen Life", "Friendship", "The Internet"]

for topic in topics:
    response = requests.post(
        f"{BASE_URL}/api/ml/train-and-generate",
        params={"topic": topic, "training_limit": 150, "question_count": 10}
    )
    print(f"{topic}: {len(response.json()['questions'])} soru üretildi")
```

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing-feature`)
5. Pull Request açın

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.