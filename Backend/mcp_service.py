import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from database import db
from dotenv import load_dotenv

load_dotenv()

class GeminiService:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY bulunamadı!")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
    
    def get_similar_questions(self, topic: str = None, limit: int = 50) -> List[Dict]:
        """Belirli konudan veya tüm sorulardan benzer soruları getir - HIZLANDIRILMIŞ"""
        try:
            if topic:
                # Daha basit sorgu (hızlandırma)
                query = """
                    SELECT id, year, question_text, option_a, option_b, option_c, option_d, 
                           correct_option, topic
                    FROM lgs_questions 
                    WHERE topic ILIKE %s
                    ORDER BY year DESC 
                    LIMIT %s
                """
                questions = db.execute_query(query, (f"%{topic}%", limit))
            else:
                query = """
                    SELECT id, year, question_text, option_a, option_b, option_c, option_d, 
                           correct_option, topic
                    FROM lgs_questions 
                    ORDER BY year DESC 
                    LIMIT %s
                """
                questions = db.execute_query(query, (limit,))
            
            return [dict(q) for q in questions] if questions else []
            
        except Exception as e:
            print(f"Sorular getirilirken hata: {e}")
            return []
    
    def get_topic_distribution(self) -> Dict[str, Any]:
        """Konu dağılımını analiz et - CACHE'DEN AL"""
        try:
            # Önce cache tablosundan dene
            query = """
                SELECT topic, total_questions as count, overall_percentage as percentage
                FROM lgs_topic_summary
                ORDER BY total_questions DESC
            """
            results = db.execute_query(query)
            
            if results:
                # Cache'den başarıyla alındı
                distribution = {}
                for row in results:
                    row_dict = dict(row)
                    distribution[row_dict['topic']] = {
                        'count': row_dict['count'],
                        'percentage': float(row_dict['percentage'])
                    }
                return distribution
            
            # Cache yoksa fallback
            print("⚠️ Cache bulunamadı, fallback kullanılıyor...")
            query = """
                SELECT topic, COUNT(*) as count, 
                       ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
                FROM lgs_questions 
                WHERE year >= 2020
                GROUP BY topic 
                ORDER BY count DESC
                LIMIT 10
            """
            results = db.execute_query(query)
            
            if not results:
                return {}
            
            distribution = {}
            for row in results:
                row_dict = dict(row)
                distribution[row_dict['topic']] = {
                    'count': row_dict['count'],
                    'percentage': float(row_dict['percentage'])
                }
            
            return distribution
            
        except Exception as e:
            print(f"Konu dağılımı analizi hatası: {e}")
            return {}
    
    def get_cached_patterns(self) -> Dict[str, Any]:
        """Cache'den kalıpları al - ÇOK HIZLI"""
        try:
            # Yıl bazında özet
            year_query = """
                SELECT year, SUM(question_count) as count
                FROM lgs_statistics 
                GROUP BY year 
                ORDER BY year DESC
                LIMIT 10
            """
            year_results = db.execute_query(year_query)
            
            # Konu bazında özet
            topic_query = """
                SELECT topic, total_questions as count, overall_percentage as percentage
                FROM lgs_topic_summary
                ORDER BY total_questions DESC
                LIMIT 15
            """
            topic_results = db.execute_query(topic_query)
            
            # Toplam soru sayısı
            total_query = "SELECT SUM(total_questions) as total FROM lgs_topic_summary"
            total_result = db.execute_single_query(total_query)
            
            # Sonuçları formatla
            years = {}
            if year_results:
                for row in year_results:
                    row_dict = dict(row)
                    years[row_dict['year']] = row_dict['count']
            
            topics = {}
            if topic_results:
                for row in topic_results:
                    row_dict = dict(row)
                    topics[row_dict['topic']] = {
                        'count': row_dict['count'],
                        'percentage': float(row_dict['percentage'])
                    }
            
            total_questions = total_result['total'] if total_result else 0
            
            return {
                'total_questions': total_questions,
                'topics': topics,
                'years': years,
                'unique_topics': len(topics)
            }
            
        except Exception as e:
            print(f"Cache analiz hatası: {e}")
            return {
                'total_questions': 0,
                'topics': {},
                'years': {},
                'unique_topics': 0
            }
    
    def predict_question_distribution(self, target_year: int = 2025) -> Dict[str, Any]:
        """Gelecek yıl için soru dağılımını tahmin et - CACHE KULLANARAK SÜPER HIZLI"""
        try:
            # Cache'den kalıpları al (çok hızlı)
            patterns = self.get_cached_patterns()
            
            if not patterns or patterns['total_questions'] == 0:
                return {"error": "Cache verisi bulunamadı - lütfen create_statistics_table.py çalıştırın"}
            
            # Çok kısa prompt (hızlandırma)
            prompt = f"""Predict {target_year} LGS English questions:

Historical data:
- Total: {patterns['total_questions']} questions
- Top topics: {json.dumps(dict(list(patterns['topics'].items())[:5]))}

Return JSON:
{{
  "year": {target_year},
  "predicted_topics": {{
    "topic_name": {{"predicted_count": number, "probability": percent}}
  }},
  "total_predicted": total_number,
  "confidence": percent
}}"""
            
            # Hızlı config
            generation_config = {
                'temperature': 0.1,
                'top_p': 0.7,
                'top_k': 10,
                'max_output_tokens': 400,
            }
            
            response = self.model.generate_content(prompt, generation_config=generation_config)
            
            # Hızlı parse
            try:
                text = response.text.strip()
                if '```' in text:
                    text = text.split('```')[1] if text.count('```') >= 2 else text.replace('```', '')
                
                start = text.find('{')
                end = text.rfind('}') + 1
                if start != -1 and end > start:
                    result = json.loads(text[start:end])
                    result['cache_used'] = True
                    return result
                
                return {"error": "JSON bulunamadı", "raw": text[:100]}
                
            except Exception as parse_error:
                return {"error": f"Parse hatası: {str(parse_error)}"}
                
        except Exception as e:
            return {"error": f"Tahmin hatası: {str(e)}"}
    
    def generate_questions(self, topic: str, count: int = 5, difficulty: str = "orta") -> List[Dict[str, Any]]:
        """Belirli konuda yeni sorular üret - HIZLANDIRILMIŞ"""
        try:
            # Daha az örnek soru al (hızlandırma)
            similar_questions = self.get_similar_questions(topic=topic, limit=5)
            
            if not similar_questions:
                return [{"error": f"'{topic}' konusunda örnek soru bulunamadı"}]
            
            # Sadece 2 örnek kullan (hızlandırma)
            examples = []
            for q in similar_questions[:2]:
                examples.append({
                    "question": q['question_text'][:100] + "...",  # Kısa tut
                    "correct": q['correct_option']
                })
            
            # Çok kısa prompt (hızlandırma)
            prompt = f"""Create {count} LGS English questions about "{topic}". Difficulty: {difficulty}

Examples: {json.dumps(examples, ensure_ascii=False)}

Return JSON:
{{
  "questions": [
    {{
      "question_text": "question with blank ----",
      "option_a": "A",
      "option_b": "B", 
      "option_c": "C",
      "option_d": "D",
      "correct_option": "A",
      "explanation": "short reason"
    }}
  ]
}}"""
            
            # Hızlı generation config
            generation_config = {
                'temperature': 0.3,  # Daha deterministik
                'top_p': 0.8,
                'top_k': 20,  # Daha az seçenek
                'max_output_tokens': 1000,  # Daha kısa
            }
            
            response = self.model.generate_content(prompt, generation_config=generation_config)
            
            # Hızlı JSON parse
            try:
                text = response.text.strip()
                # Basit temizleme
                if '```' in text:
                    text = text.split('```')[1] if text.count('```') >= 2 else text.replace('```', '')
                
                # JSON bul ve parse et
                start = text.find('{')
                end = text.rfind('}') + 1
                if start != -1 and end > start:
                    json_text = text[start:end]
                    generated = json.loads(json_text)
                    return generated.get('questions', [])
                
                return [{"error": "JSON bulunamadı"}]
                
            except Exception as parse_error:
                return [{"error": f"Parse hatası: {str(parse_error)}", "raw": response.text[:200]}]
                
        except Exception as e:
            return [{"error": f"Soru üretme hatası: {str(e)}"}]
    
    def generate_mixed_questions(self, total_count: int = 10) -> List[Dict[str, Any]]:
        """Tüm konulardan orantılı soru üret - CACHE KULLANARAK SÜPER HIZLI"""
        try:
            # Cache'den en popüler konuları al
            query = """
                SELECT topic, total_questions
                FROM lgs_topic_summary
                ORDER BY total_questions DESC 
                LIMIT 3
            """
            topic_results = db.execute_query(query)
            
            if not topic_results:
                return [{"error": "Cache'de konu bulunamadı - lütfen create_statistics_table.py çalıştırın"}]
            
            # Orantılı dağılım hesapla
            topics_data = [dict(t) for t in topic_results]
            total_historical = sum([t['total_questions'] for t in topics_data])
            
            all_questions = []
            remaining_count = total_count
            
            for i, topic_data in enumerate(topics_data):
                if remaining_count <= 0:
                    break
                
                topic = topic_data['topic']
                
                # Orantılı hesaplama
                if i == len(topics_data) - 1:  # Son konu
                    count = remaining_count
                else:
                    proportion = topic_data['total_questions'] / total_historical
                    count = max(1, int(total_count * proportion))
                    count = min(count, remaining_count)
                
                if count > 0:
                    topic_questions = self.generate_questions(topic, count, "orta")
                    
                    # Başarılı soruları ekle
                    for q in topic_questions:
                        if 'error' not in q:
                            q['question_id'] = len(all_questions) + 1
                            q['cache_generated'] = True
                            all_questions.append(q)
                    
                    remaining_count -= count
            
            return all_questions[:total_count]
            
        except Exception as e:
            return [{"error": f"Karma soru üretme hatası: {str(e)}"}]

# Global service instance
gemini_service = GeminiService()