# main.py
from fastapi import FastAPI

# 1. FastAPI uygulamasını başlat
# Projenin başlığını ve açıklamasını şimdiden belirleyebilirsiniz
app = FastAPI(
    title="LGS Soru Üretici API",
    description="Geçmiş LGS sorularından öğrenerek yeni sorular üreten API.",
    version="0.0.1"
)

# 2. API'nin çalışıp çalışmadığını test etmek için "root" endpoint'i
# Tarayıcıda http://127.0.0.1:8000/ adresine girince bu görünür
@app.get("/")
def read_root():
    return {"mesaj": "LGS Soru Üretici API - Sunucu Ayakta!"}


# 3. Gelecekteki soru üretme endpoint'i için bir yer tutucu (placeholder)
# Şimdilik yorum satırı olarak kalabilir veya basit bir mesaj dönebilir
@app.post("/uret/")
def placeholder_uret():
    # Modeliniz hazır olduğunda bu fonksiyonu dolduracaksınız
    return {"mesaj": "Soru üretme endpoint'i - Henüz aktif değil."}