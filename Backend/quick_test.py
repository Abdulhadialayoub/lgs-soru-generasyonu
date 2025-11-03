"""
Hızlı ML Test Script
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_train():
    """Model eğitimini test et"""
    print("🚀 Model eğitimi başlıyor...")
    
    response = requests.post(
        f"{BASE_URL}/api/ml/train",
        params={"limit": 100}
    )
    
    print(f"Status: {response.status_code}")
    
    data = response.json()
    print(json.dumps(data, indent=2, ensure_ascii=False))
    
    if data.get('success'):
        print("\n✅ Model başarıyla eğitildi!")
        stats = data['stats']
        print(f"Model Type: {stats.get('model_type')}")
        print(f"Accuracy: {stats.get('topic_accuracy')}%")
        print(f"Top Features: {stats.get('top_features', [])[:5]}")
    else:
        print(f"\n❌ Hata: {data.get('message')}")

if __name__ == "__main__":
    test_train()
