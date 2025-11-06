# LGS Soru Üretim Sistemi - Frontend

Modern React + Vite ile geliştirilmiş LGS İngilizce soru üretim ve analiz platformu.

## 🚀 Özellikler

- **Dashboard**: Genel istatistikler ve grafikler
- **Soru Bankası**: Geçmiş LGS sorularını görüntüleme ve filtreleme
- **MCP Soru Üretimi**: Model Context Protocol ile yeni soru üretimi
- **Sınav Oluşturma**: Gerçekçi LGS sınavı oluşturma
- **ML Model**: Machine Learning ile soru üretimi ve model eğitimi
- **İstatistikler**: Detaylı analiz ve grafikler

## 📦 Kurulum

### 1. Bağımlılıkları Yükleyin

```bash
npm install
```

### 2. Ortam Değişkenlerini Ayarlayın

`.env` dosyasını düzenleyin:

```env
VITE_API_URL=http://localhost:8000
```

### 3. Geliştirme Sunucusunu Başlatın

```bash
npm run dev
```

Uygulama `http://localhost:5173` adresinde çalışacaktır.

## 🏗️ Build

Production build için:

```bash
npm run build
```

Build dosyaları `dist/` klasöründe oluşturulacaktır.

## 📁 Proje Yapısı

```
Frontend/
├── src/
│   ├── components/        # Yeniden kullanılabilir bileşenler
│   │   ├── Layout.jsx
│   │   └── Sidebar.jsx
│   ├── pages/            # Sayfa bileşenleri
│   │   ├── Dashboard.jsx
│   │   ├── QuestionBank.jsx
│   │   ├── GenerateQuestions.jsx
│   │   ├── CreateExam.jsx
│   │   ├── MLModel.jsx
│   │   └── Statistics.jsx
│   ├── config/           # Yapılandırma dosyaları
│   │   └── api.js        # API servisleri
│   ├── App.jsx           # Ana uygulama
│   ├── App.css           # Global stiller
│   └── main.jsx          # Giriş noktası
├── public/               # Statik dosyalar
├── .env                  # Ortam değişkenleri
├── package.json
└── vite.config.js
```

## 🎨 Teknolojiler

- **React 19** - UI framework
- **Vite** - Build tool
- **React Router** - Routing
- **Axios** - HTTP client
- **Recharts** - Grafikler
- **Lucide React** - İkonlar

## 🔌 API Entegrasyonu

Backend API'si ile tam entegrasyon:

- Soru yönetimi
- İstatistik verileri
- MCP soru üretimi
- ML model eğitimi ve üretimi
- Sınav oluşturma

## 📱 Responsive Tasarım

Tüm ekran boyutlarında çalışır:
- Desktop (1400px+)
- Tablet (768px - 1024px)
- Mobile (< 768px)

## 🎯 Kullanım

### Dashboard
- Genel istatistikleri görüntüleyin
- Konu dağılımı grafiklerini inceleyin
- ML model durumunu kontrol edin

### Soru Bankası
- Geçmiş LGS sorularını filtreleyin
- Yıl ve konuya göre arama yapın
- Soruları detaylı inceleyin

### Soru Üret
- Konu seçin (veya karma)
- Soru sayısı ve zorluk belirleyin
- AI ile yeni sorular üretin

### Sınav Oluştur
- Soru sayısı belirleyin (5-20)
- Gerçekçi LGS sınavı oluşturun
- Sınavı .txt formatında indirin

### ML Model
- Model eğitimi yapın
- Eğitilmiş modelle soru üretin
- Model durumunu görüntüleyin

### İstatistikler
- Detaylı analiz grafikleri
- Konu dağılımı
- Yıllara göre trend analizi

## 🔧 Geliştirme

```bash
# Geliştirme sunucusu
npm run dev

# Lint kontrolü
npm run lint

# Build
npm run build

# Preview (build sonrası)
npm run preview
```

## 📝 Notlar

- Backend API'sinin çalışıyor olması gerekir
- CORS ayarları backend'de yapılandırılmıştır
- Tüm API istekleri axios ile yönetilir

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push yapın (`git push origin feature/amazing`)
5. Pull Request açın

## 📄 Lisans

MIT License
