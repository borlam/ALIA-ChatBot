# -*- coding: utf-8 -*-
"""Motor de chat MEJORADO con corrección automática y precisión histórica"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict
from datetime import datetime
import re
from ..system.config import *

class ChatEngine:
    def __init__(self):
        print("🧠 Cargando modelo salamandra-7b (modo optimizado)...")
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=False,
            low_cpu_mem_usage=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"✅ Modelo 7B cargado en modo optimizado")
        
        # Diccionario de correcciones comunes
        self.common_corrections = {
            "treces colonias": "trece colonias",
            "las treces colinias": "las trece colonias",
            "eslavos": "esclavos",
            "hispanoamericanavirreinal": "hispanoamericana virreinal",
            "virreinal": "virreinal",
            "conquista": "conquista",
            "descubrimiento": "descubrimiento",
            "moctezuma": "Moctezuma",
            "cortés": "Cortés",
            "inca": "inca",
            "azteca": "azteca",
            "maya": "maya"
        }
    
    def correct_question(self, question: str) -> str:
        """Corrige automáticamente errores comunes en la pregunta"""
        corrected = question.lower()
        
        for wrong, right in self.common_corrections.items():
            if wrong in corrected:
                # Usar regex para reemplazar manteniendo mayúsculas iniciales
                corrected = re.sub(
                    re.escape(wrong), 
                    lambda m: right if m.group().islower() else right.title(),
                    corrected,
                    flags=re.IGNORECASE
                )
        
        return corrected
    
    def extract_key_concepts(self, question: str) -> List[str]:
        """Extrae conceptos clave para búsqueda contextual"""
        question_lower = question.lower()
        concepts = []
        
        # Conceptos históricos
        historical_terms = [
            'esclavos', 'esclavitud', 'esclavo fugitivo',
            'trece colonias', 'colonias británicas',
            'españa', 'territorios españoles', 'florida', 'luisiana',
            'libertad', 'asilo', 'decreto', 'ley',
            'virreinal', 'colonial', 'conquista', 'descubrimiento',
            'cimarrones', 'libertos', 'fort mose',
            'siglo xvii', 'siglo xviii', '1693', '1738'
        ]
        
        for term in historical_terms:
            if term in question_lower:
                concepts.append(term)
        
        # Extraer nombres propios (capitalizados)
        words = question.split()
        for word in words:
            if word[0].isupper() and len(word) > 2:
                concepts.append(word.lower())
        
        return list(set(concepts))
    
    def build_intelligent_context(self, question: str, documents: List[Dict]) -> str:
        """
        Construye contexto INTELIGENTE priorizando documentos relevantes
        """
        if not documents:
            return "No hay documentos relevantes para esta pregunta."
        
        # Extraer conceptos clave de la pregunta
        key_concepts = self.extract_key_concepts(question)
        
        # Clasificar documentos por relevancia
        scored_docs = []
        for doc in documents:
            score = 0
            doc_text_lower = doc['text'].lower()
            
            # Puntuar por conceptos clave
            for concept in key_concepts:
                if concept in doc_text_lower:
                    score += 3  # Concepto principal
                elif any(word in doc_text_lower for word in concept.split()):
                    score += 1  # Palabra relacionada
            
            # Bonus por metadatos enriquecidos
            enriched = doc.get('enriched_metadata', {})
            if enriched.get('themes'):
                doc_themes_lower = [t.lower() for t in enriched['themes']]
                for concept in key_concepts:
                    if any(concept in theme for theme in doc_themes_lower):
                        score += 2
            
            scored_docs.append((score, doc))
        
        # Ordenar por relevancia
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Construir contexto
        context_parts = []
        context_parts.append("### 📚 INFORMACIÓN DOCUMENTAL RELEVANTE:")
        
        for i, (score, doc) in enumerate(scored_docs[:3]):  # Top 3 más relevantes
            if score < 1:  # Si no es muy relevante, omitir
                continue
                
            source = doc.get('pdf_title', doc.get('filename', f'Documento {i+1}'))
            text_preview = self._extract_most_relevant_snippet(doc['text'], key_concepts)
            
            context_parts.append(f"\n**{source}** (relevancia: {score}/10)")
            
            if text_preview:
                context_parts.append(f"*Extracto relevante:* {text_preview}")
            
            # Añadir metadatos enriquecidos si existen
            enriched = doc.get('enriched_metadata', {})
            if enriched.get('themes'):
                themes = enriched['themes'][:3]
                context_parts.append(f"*Temas:* {', '.join(themes)}")
            
            if enriched.get('key_dates'):
                dates = enriched['key_dates'][:2]
                context_parts.append(f"*Fechas clave:* {', '.join(dates)}")
            
            if i < len(scored_docs[:3]) - 1:
                context_parts.append("---")
        
        # Si no hay documentos muy relevantes
        if len(context_parts) == 1:  # Solo el título
            context_parts.append("\n(No se encontraron documentos altamente relevantes)")
        
        return '\n'.join(context_parts)
    
    def _extract_most_relevant_snippet(self, text: str, key_concepts: List[str]) -> str:
        """Extrae el fragmento más relevante del texto"""
        sentences = re.split(r'[.!?]+', text)
        
        best_sentence = ""
        best_score = 0
        
        for sentence in sentences:
            if len(sentence) < 20 or len(sentence) > 300:
                continue
                
            sentence_lower = sentence.lower()
            score = 0
            
            # Puntuar por conceptos clave
            for concept in key_concepts:
                if concept in sentence_lower:
                    score += 3
                elif any(word in sentence_lower for word in concept.split()):
                    score += 1
            
            if score > best_score:
                best_score = score
                best_sentence = sentence.strip()
        
        if best_sentence:
            return best_sentence[:250] + ("..." if len(best_sentence) > 250 else "")
        
        # Fallback: primeras frases
        for sentence in sentences:
            if len(sentence) >= 50:
                return sentence[:250] + "..." if len(sentence) > 250 else sentence
        
        return text[:250] + "..." if len(text) > 250 else text
    
    def build_optimized_prompt(self, question: str, context: str) -> str:
        """Prompt optimizado para precisión histórica"""
        
        corrected_question = self.correct_question(question)
        
        return f"""Eres regerIA, un historiador especialista en historia colonial hispanoamericana con amplio conocimiento documental.

<CONTEXTO_DOCUMENTAL>
{context}
</CONTEXTO_DOCUMENTAL>

<INSTRUCCIONES_DETALLADAS>
1. **Precisión histórica**: Usa fechas, lugares y nombres específicos cuando sean relevantes
2. **Corrección automática**: Si detectas errores en la pregunta, corrígelos sutilmente en tu respuesta
3. **Citar fuentes**: Cuando uses información específica de los documentos, menciónalo implícitamente
4. **Estructura clara**: 
   - Comienza con una respuesta directa a la pregunta
   - Proporciona contexto histórico relevante
   - Incluye ejemplos específicos cuando sea posible
   - Concluye con el significado histórico del tema
5. **Evitar etiquetas**: No uses "Conclusión:", "Análisis:" como secciones separadas
6. **Concisión**: Sé informativo pero conciso (400-600 palabras)
</INSTRUCCIONES_DETALLADAS>

<FORMATO_DE_RESPUESTA>
- Párrafo inicial: Respuesta directa y clara
- Párrafos centrales: Contexto, ejemplos, detalles específicos
- Párrafo final: Significado histórico y legado
</FORMATO_DE_RESPUESTA>

<CORRECCIONES_APLICABLES>
Pregunta original: "{question}"
Pregunta corregida: "{corrected_question}"
</CORRECCIONES_APLICABLES>

PREGUNTA A RESPONDER: {corrected_question}

RESPUESTA DEL HISTORIADOR:"""
    
    def post_process_response(self, response: str, question: str) -> str:
        """Post-procesa la respuesta para mejorar calidad"""
        
        # 1. Eliminar repeticiones del prompt
        if "RESPUESTA DEL HISTORIADOR:" in response:
            response = response.split("RESPUESTA DEL HISTORIADOR:")[-1].strip()
        
        # 2. Limpiar marcas XML/HTML
        response = re.sub(r'</?[^>]+>', '', response)
        
        # 3. Unificar espacios
        response = re.sub(r'\s+', ' ', response)
        
        # 4. Corregir errores comunes en la respuesta
        for wrong, right in self.common_corrections.items():
            if wrong in response.lower():
                # Reemplazar manteniendo capitalización
                pattern = re.compile(re.escape(wrong), re.IGNORECASE)
                response = pattern.sub(
                    lambda m: right if m.group().islower() else right.title(),
                    response
                )
        
        # 5. Asegurar puntuación final
        if response and response[-1] not in '.!?':
            response = response + '.'
        
        # 6. Limitar longitud razonable
        if len(response) > 1200:
            # Buscar un punto de corte natural
            sentences = re.split(r'[.!?]+', response)
            trimmed = ""
            char_count = 0
            for sentence in sentences:
                if char_count + len(sentence) < 1100:
                    trimmed += sentence + '.'
                    char_count += len(sentence) + 1
                else:
                    break
            if trimmed:
                response = trimmed.strip()
        
        return response.strip()
    
    def generate_response(self, question: str, context_docs: List[Dict], max_chars: int = 1200) -> str:
        """Genera respuesta MEJORADA con precisión histórica"""
        
        start_time = datetime.now()
        
        print(f"🤔 Pregunta: {question}")
        
        # 1. Construir contexto inteligente
        context = self.build_intelligent_context(question, context_docs)
        
        # 2. Construir prompt optimizado
        prompt = self.build_optimized_prompt(question, context)
        
        # 3. Tokenización
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2800
        ).to(self.model.device)
        
        # 4. Generación con parámetros optimizados para calidad
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=800,
                temperature=0.7,
                do_sample=True,
                top_p=0.92,
                top_k=50,
                repetition_penalty=1.18,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 5. Decodificar y post-procesar
        raw_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = self.post_process_response(raw_response, question)
        
        # 6. Estadísticas
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"✅ Respuesta en {elapsed:.1f}s, {len(response)} caracteres")
        print(f"📊 Tokens de entrada: {inputs['input_ids'].shape[1]}")
        
        # 7. Limpiar memoria
        self.cleanup_memory()
        
        return response
    
    def cleanup_memory(self):
        """Limpia memoria GPU"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_response_quality_score(self, response: str, question: str) -> Dict[str, float]:
        """Evalúa la calidad de una respuesta"""
        
        score = 0
        max_score = 10
        feedback = []
        
        # 1. Longitud adecuada
        if 300 <= len(response) <= 1200:
            score += 2
        else:
            feedback.append("Longitud inadecuada")
        
        # 2. Estructura de párrafos
        paragraphs = response.split('\n\n')
        if len(paragraphs) >= 2:
            score += 2
        else:
            feedback.append("Falta estructura en párrafos")
        
        # 3. Precisión terminológica
        precision_indicators = ['siglo', 'año', 'decreto', 'ley', 'política', 'histórico']
        precision_count = sum(1 for indicator in precision_indicators if indicator in response.lower())
        if precision_count >= 2:
            score += 3
        else:
            feedback.append("Falta precisión histórica")
        
        # 4. Sin errores comunes
        error_count = sum(1 for wrong in self.common_corrections if wrong in response.lower())
        if error_count == 0:
            score += 3
        else:
            feedback.append(f"Contiene {error_count} errores comunes")
        
        return {
            'score': score / max_score,
            'percentage': (score / max_score) * 100,
            'feedback': feedback if feedback else ["Buena calidad"]
        }