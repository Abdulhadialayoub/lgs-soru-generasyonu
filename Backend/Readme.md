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

- **GET /** - Ana sayfa
- **GET /api/questions** - Tüm soruları getir (filtreleme seçenekleri ile)
- **GET /api/questions/{id}** - Belirli bir soruyu getir
- **GET /api/stats** - Soru istatistikleri

## Swagger Dokümantasyonu

Uygulama çalıştıktan sonra şu adreslerde API dokümantasyonuna erişebilirsiniz:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Veritabanı

Proje Supabase PostgreSQL kullanmaktadır. Veritabanı bağlantı bilgileri `.env` dosyasında tanımlanmıştır.
