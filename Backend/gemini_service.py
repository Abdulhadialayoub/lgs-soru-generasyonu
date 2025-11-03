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
        # Optional feature flag for future cached content use
        self.use_cached_content = os.getenv("USE_GEMINI_CACHE", "false").lower() == "true"
        # Stable system instruction to standardize outputs
        self.SYSTEM_INSTRUCTION = (
            "Sen bir LGS İngilizce soru üretim uzmanısın. LGS seviyesinde, tek doğru cevaplı, "
            "A/B/C/D seçenekli sorular üret. Çıktıyı her zaman geçerli JSON olarak ver. "
            "Şema: { 'questions': [ { 'question_text', 'option_a', 'option_b', 'option_c', 'option_d', 'correct_option', 'explanation' } ] }."
        )
    
    def get_similar_questions(self, topic: str = None, limit: int = 50) -> List[Dict]:
        """Belirli konudan veya tüm sorulardan benzer soruları getir"""
        try:
            if topic:
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
    
    def analyze_question_patterns(self, questions: List[Dict]) -> Dict[str, Any]:
        """Soru kalıplarını analiz et"""
        try:
            topics = {}
            years = {}
            
            for q in questions:
                topic = q['topic']
                year = q['year']
                
                if topic not in topics:
                    topics[topic] = []
                topics[topic].append(q)
                
                if year not in years:
                    years[year] = 0
                years[year] += 1
            
            topic_stats = {}
            for topic, topic_questions in topics.items():
                topic_stats[topic] = {
                    'count': len(topic_questions),
                    'percentage': round((len(topic_questions) / len(questions)) * 100, 2),
                    'recent_years': list(set([q['year'] for q in topic_questions[-5:]]))
                }
            
            return {
                'total_questions': len(questions),
                'topics': topic_stats,
                'years': dict(sorted(years.items(), reverse=True)),
                'unique_topics': len(topics)
            }
            
        except Exception as e:
            print(f"Analiz hatası: {e}")
            return {}
    
    def predict_question_distribution(self, target_year: int = 2025) -> Dict[str, Any]:
        """Gelecek yıl için soru dağılımını tahmin et"""
        try:
            questions = self.get_similar_questions(limit=200)
            
            if not questions:
                return {"error": "Yeterli veri bulunamadı"}
            
            patterns = self.analyze_question_patterns(questions)
            
            prompt = f"""
            LGS İngilizce soru verilerini analiz ederek {target_year} yılı için tahmin yap.

            Mevcut veriler:
            - Toplam soru sayısı: {patterns['total_questions']}
            - Konu dağılımı: {json.dumps(patterns['topics'], indent=2)}

            Lütfen {target_year} yılı için şunları tahmin et:
            1. Her konudan kaç soru gelebileceği
            2. Hangi konuların daha çok çıkma ihtimali olduğu
            3. Toplam soru sayısı tahmini

            Cevabını JSON formatında ver:
            {{
                "year": {target_year},
                "predicted_topics": {{
                    "konu_adı": {{
                        "predicted_count": sayı,
                        "probability": yüzde,
                        "trend": "artış/azalış/sabit"
                    }}
                }},
                "total_predicted": toplam_sayı,
                "confidence": güven_yüzdesi,
                "analysis": "kısa_analiz"
            }}
            """
            
            response = self.model.generate_content(prompt)
            
            try:
                # Response text'i güvenli şekilde al
                response_text = ""
                if hasattr(response, 'text'):
                    response_text = response.text
                elif hasattr(response, 'parts') and response.parts:
                    response_text = response.parts[0].text
                elif hasattr(response, 'candidates') and response.candidates:
                    if response.candidates[0].content.parts:
                        response_text = response.candidates[0].content.parts[0].text
                
                if not response_text:
                    return {"error": "Gemini'den boş cevap geldi"}
                
                # JSON parse et
                clean_text = response_text.strip().replace('```json', '').replace('```', '')
                prediction = json.loads(clean_text)
                return prediction
                
            except json.JSONDecodeError:
                return {
                    "year": target_year,
                    "error": "JSON parse edilemedi",
                    "raw_response": response_text[:200] if response_text else "Boş response"
                }
                
        except Exception as e:
            return {"error": f"Tahmin hatası: {str(e)}"}
    
    def _format_vector_for_sql(self, vector: List[float]) -> str:
        """pgvector için Python list -> '[...]' string formatı"""
        return "[" + ",".join(f"{float(v):.6f}" for v in vector) + "]"

    def embed_text(self, text: str) -> Optional[List[float]]:
        """Metni embedding vektörüne çevir (text-embedding-004)."""
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="RETRIEVAL_QUERY"
            )
            # SDK bazen dict, bazen dot-access dönebilir; güvenli erişim
            if isinstance(result, dict) and 'embedding' in result:
                return result['embedding']
            if hasattr(result, 'embedding'):
                return getattr(result, 'embedding')
            return None
        except Exception:
            return None

    def get_similar_by_embedding(self, embedding: List[float], k: int = 8) -> List[Dict[str, Any]]:
        """pgvector mesafesine göre en benzer k soruyu getir."""
        try:
            if not db.connection:
                db.connect()
            # embedding parametresi vector cast ile gönderilir
            query = """
                SELECT id, year, question_text, option_a, option_b, option_c, option_d,
                       correct_option, topic
                FROM lgs_questions
                WHERE embedding IS NOT NULL
                ORDER BY embedding <-> %s::vector
                LIMIT %s
            """
            vec_param = self._format_vector_for_sql(embedding)
            rows = db.execute_query(query, (vec_param, k))
            return [dict(r) for r in rows] if rows else []
        except Exception:
            return []

    def get_similar_by_text(self, text: str, k: int = 8) -> List[Dict[str, Any]]:
        """Text'i embed edip pgvector ile benzer soruları getir; başarısızsa konu LIKE fallback."""
        # 1) Embedding tabanlı arama
        embedding = self.embed_text(text)
        if embedding:
            results = self.get_similar_by_embedding(embedding, k)
            if results:
                return results
        # 2) Fallback: mevcut LIKE tabanlı sorgu
        return self.get_similar_questions(topic=text, limit=k)

    def generate_questions(self, topic: str, count: int = 5, difficulty: str = "orta") -> List[Dict[str, Any]]:
        """Belirli konuda yeni sorular üret"""
        try:
            # Önce embedding tabanlı benzer örnekleri getir (RAG)
            similar_questions = self.get_similar_by_text(topic, k=10)
            
            if not similar_questions:
                return [{"error": f"'{topic}' konusunda örnek soru bulunamadı"}]
            
            # Few-shot örnekleri (3 adet, kısa ve öz)
            examples = []
            for q in similar_questions[:3]:
                examples.append({
                    "question": q['question_text'],
                    "options": {
                        "A": q['option_a'],
                        "B": q['option_b'], 
                        "C": q['option_c'],
                        "D": q['option_d']
                    },
                    "correct": q['correct_option']
                })
            
            # Sistem talimatı + dinamik kısım (örnekler dahil)
            prompt = (
                self.SYSTEM_INSTRUCTION
                + "\n\nGörev: '{topic}' konusunda {count} adet yeni soru üret. Zorluk: {difficulty}.\n".format(
                    topic=topic, count=count, difficulty=difficulty
                )
                + "\nÖrnek sorular (referans):\n"
                + json.dumps(examples, indent=2, ensure_ascii=False)
                + "\nKurallar:\n"
                  "1) 4 seçenek A/B/C/D, 2) Tek doğru cevap, 3) LGS seviyesi, 4) Konuya uygun.\n"
                + "Çıktı şeması:\n{\n  \"questions\": [ { \"question_text\", \"option_a\", \"option_b\", \"option_c\", \"option_d\", \"correct_option\", \"explanation\" } ]\n}\n"
            )
            
            response = self.model.generate_content(prompt)
            
            try:
                # Response text'i güvenli şekilde al
                response_text = ""
                if hasattr(response, 'text'):
                    response_text = response.text
                elif hasattr(response, 'parts') and response.parts:
                    response_text = response.parts[0].text
                elif hasattr(response, 'candidates') and response.candidates:
                    if response.candidates[0].content.parts:
                        response_text = response.candidates[0].content.parts[0].text
                
                if not response_text:
                    return [{"error": "Gemini'den boş cevap geldi"}]
                
                # JSON parse et
                clean_text = response_text.strip().replace('```json', '').replace('```', '')
                generated = json.loads(clean_text)
                return generated.get('questions', [])
                
            except json.JSONDecodeError:
                return [{
                    "error": "JSON parse edilemedi",
                    "raw_response": response_text[:200] if response_text else "Boş response"
                }]
                
        except Exception as e:
            return [{"error": f"Soru üretme hatası: {str(e)}"}]
    
    def generate_mixed_questions(self, total_count: int = 10) -> List[Dict[str, Any]]:
        """Tüm konulardan orantılı soru üret"""
        try:
            # En popüler konuları al
            query = """
                SELECT topic, COUNT(*) as count 
                FROM lgs_questions 
                GROUP BY topic 
                ORDER BY count DESC 
                LIMIT 3
            """
            topic_results = db.execute_query(query)
            
            if not topic_results:
                return [{"error": "Konu bulunamadı"}]
            
            topics = [dict(t)['topic'] for t in topic_results]
            questions_per_topic = total_count // len(topics)
            
            all_questions = []
            
            for i, topic in enumerate(topics):
                count = questions_per_topic
                if i == 0:  # İlk konuya kalan soruları da ver
                    count += total_count % len(topics)
                
                if count > 0:
                    topic_questions = self.generate_questions(topic, count, "orta")
                    
                    for q in topic_questions:
                        if 'error' not in q:
                            q['question_id'] = len(all_questions) + 1
                            all_questions.append(q)
            
            return all_questions[:total_count]
            
        except Exception as e:
            return [{"error": f"Karma soru üretme hatası: {str(e)}"}]
    
    def generate_exam_questions(self, total_count: int = 10) -> Dict[str, Any]:
        """Gerçekçi LGS sınavı üret - Akıllı analiz ile"""
        try:
            # İstatistik tablosundan ağırlıkları al
            topic_weights = self._get_topic_weights_from_stats()
            
            if not topic_weights:
                print("İstatistik tablosundan ağırlık alınamadı, fallback kullanılıyor...")
                return self._generate_simple_exam(total_count)
            
            # Maksimum soru sayısını hesapla
            max_per_topic = max(2, total_count // 5)
            
            # Gerçekçi dağılım hesapla
            topic_distribution = self._calculate_realistic_distribution(topic_weights, total_count, max_per_topic)
            
            # Her konu için soru üret
            all_questions = []
            exam_distribution = {}
            
            for topic, count in topic_distribution.items():
                if count > 0:
                    topic_questions = self.generate_questions(topic, count, "orta")
                    
                    successful_questions = []
                    for q in topic_questions:
                        if 'error' not in q:
                            q['question_id'] = len(all_questions) + 1
                            q['exam_topic'] = topic
                            successful_questions.append(q)
                            all_questions.append(q)
                    
                    exam_distribution[topic] = len(successful_questions)
            
            # Sınav metadatası
            exam_metadata = {
                'total_questions': len(all_questions),
                'topic_distribution': exam_distribution,
                'max_per_topic': max_per_topic,
                'exam_type': 'LGS İngilizce Gerçekçi Sınav',
                'difficulty': 'Orta',
                'estimated_time': f"{len(all_questions) * 1.5:.0f} dakika",
                'distribution_logic': 'İstatistik tablosundan otomatik hesaplanan ağırlıklar',
                'weights_source': 'lgs_statistics tablosu'
            }
            
            return {
                'exam_metadata': exam_metadata,
                'questions': all_questions,
                'success': True
            }
            
        except Exception as e:
            print(f"Sınav üretme hatası: {e}")
            return self._generate_simple_exam(total_count)
    
    def _get_topic_weights_from_stats(self) -> Dict[str, int]:
        """Hazır ağırlık tablosundan konu ağırlıklarını al"""
        try:
            # Hazır ağırlık tablosundan al
            weights_query = """
                SELECT topic, weight
                FROM topic_weights_cache 
                ORDER BY weight DESC
            """
            
            weights_results = db.execute_query(weights_query)
            
            if not weights_results:
                print("❌ Ağırlık tablosunda veri bulunamadı - lütfen calculate_topic_weights.py çalıştırın")
                return {}
            
            # Ağırlıkları dict'e çevir
            topic_weights = {}
            for weight_row in weights_results:
                weight_dict = dict(weight_row)
                topic = weight_dict['topic']
                weight = weight_dict['weight']
                topic_weights[topic] = weight
            
            print(f"✅ Hazır tablodan alınan ağırlıklar: {topic_weights}")
            return topic_weights
            
        except Exception as e:
            print(f"❌ Ağırlık tablosundan okuma hatası: {e}")
            return {}
    
    def _calculate_realistic_distribution(self, topic_weights: Dict[str, int], total_count: int, max_per_topic: int) -> Dict[str, int]:
        """Gerçekçi konu dağılımı hesapla - Ağırlıklara göre"""
        try:
            distribution = {}
            
            # Toplam ağırlığı hesapla
            total_weight = sum(topic_weights.values())
            
            # Her konu için soru sayısını hesapla
            remaining_questions = total_count
            
            # Ağırlığa göre soru sayısı hesapla
            for topic, weight in sorted(topic_weights.items(), key=lambda x: x[1], reverse=True):
                if remaining_questions <= 0:
                    break
                
                # Ağırlığa göre soru sayısı
                calculated_count = round((weight / total_weight) * total_count)
                
                # Maksimum sınırı uygula
                actual_count = min(calculated_count, max_per_topic, remaining_questions)
                
                # En az 1 soru garantisi (eğer kalan soru varsa)
                if actual_count == 0 and remaining_questions > 0 and len([k for k, v in distribution.items() if v > 0]) < 6:
                    actual_count = 1
                
                if actual_count > 0:
                    distribution[topic] = actual_count
                    remaining_questions -= actual_count
            
            # Kalan soruları en popüler konulara dağıt
            sorted_topics = sorted(topic_weights.items(), key=lambda x: x[1], reverse=True)
            
            for topic, weight in sorted_topics:
                if remaining_questions <= 0:
                    break
                
                if topic in distribution and distribution[topic] < max_per_topic:
                    add_count = min(remaining_questions, max_per_topic - distribution[topic])
                    distribution[topic] += add_count
                    remaining_questions -= add_count
                elif topic not in distribution:
                    add_count = min(remaining_questions, max_per_topic)
                    distribution[topic] = add_count
                    remaining_questions -= add_count
            
            return distribution
            
        except Exception as e:
            print(f"Gerçekçi dağılım hesaplama hatası: {e}")
            # Fallback
            return self._simple_fallback_distribution(list(topic_weights.keys()), total_count, max_per_topic)
    
    def _simple_fallback_distribution(self, topics: List[str], total_count: int, max_per_topic: int) -> Dict[str, int]:
        """Basit fallback dağılım"""
        distribution = {}
        remaining = total_count
        
        # İlk 5 konuya dağıt
        for i, topic in enumerate(topics[:5]):
            if remaining <= 0:
                break
            
            count = min(max_per_topic, remaining, total_count // 5 + (1 if i < total_count % 5 else 0))
            if count > 0:
                distribution[topic] = count
                remaining -= count
        
        return distribution
    
    def _generate_simple_exam(self, total_count: int = 10) -> Dict[str, Any]:
        """Basit sınav üretme (fallback)"""
        try:
            # Maksimum soru sayısını hesapla
            max_per_topic = max(2, total_count // 5)
            
            # Basit ağırlıklı konu listesi
            simple_weights = {
                'Teen Life': 25,
                'Friendship': 20,
                'The Internet': 15,
                'Adventures': 15,
                'Tourism': 10,
                'On The Phone': 8,
                'Science': 7
            }
            
            # Gerçekçi dağılım hesapla
            distribution = self._calculate_realistic_distribution(simple_weights, total_count, max_per_topic)
            
            all_questions = []
            exam_distribution = {}
            
            for topic, count in distribution.items():
                if count > 0:
                    topic_questions = self.generate_questions(topic, count, "orta")
                    
                    successful_questions = []
                    for q in topic_questions:
                        if 'error' not in q:
                            q['question_id'] = len(all_questions) + 1
                            q['exam_topic'] = topic
                            successful_questions.append(q)
                            all_questions.append(q)
                    
                    exam_distribution[topic] = len(successful_questions)
            
            exam_metadata = {
                'total_questions': len(all_questions),
                'topic_distribution': exam_distribution,
                'max_per_topic': max_per_topic,
                'exam_type': 'LGS İngilizce Gerçekçi Sınav (Fallback)',
                'difficulty': 'Orta',
                'estimated_time': f"{len(all_questions) * 1.5:.0f} dakika"
            }
            
            return {
                'exam_metadata': exam_metadata,
                'questions': all_questions,
                'success': True
            }
            
        except Exception as e:
            return {
                'exam_metadata': {},
                'questions': [{"error": f"Basit sınav üretme hatası: {str(e)}"}],
                'success': False
            }

# Global service instance
gemini_service = GeminiService()