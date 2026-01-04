# -*- coding: utf-8 -*-
"""Motor de chat CORREGIDO - Versión definitiva"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict, Optional
from datetime import datetime
import re
from ..system.config import *

class ChatEngine:
    def __init__(self):
        print("🧠 Cargando modelo salamandra-7b...")
        
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
        
        print(f"✅ Modelo cargado")
        
        # Errores críticos a corregir
        self.critical_corrections = {
            "eslavos": "esclavos",
            "eslavo": "esclavo",
            "esclavizadxs": "esclavizados",
            "treces colonias": "Trece Colonias",
            "colonias britanicas": "colonias británicas",
            "florida": "Florida",
            "luisiana": "Luisiana",
            "moctezuma": "Moctezuma"
        }
        
        # Patrones a eliminar (modelo hablando de sí mismo)
        self.self_reference_patterns = [
            r'Estoy (de acuerdo|conforme|satisfecho|feliz)',
            r'Me alegra',
            r'estoy feliz de',
            r'gracias a tu',
            r'compartir contigo',
            r'en mi búsqueda',
            r'entiendo mejor',
            r'he encontrado',
            r'hoy día',
            r'Ahora entiendo',
            r'Me satisface',
            r'quiero destacar que',
            r'Debo señalar que',
            r'En mi opinión personal',
            r'Creo que',
            r'Pienso que'
        ]
        
        # Estructuras problemáticas
        self.problematic_structures = [
            r'Conclusión:\s*\.',
            r'Análisis crítico:\s*\.', 
            r'Contexto histórico:\s*\.',
            r'Evidencia documental:\s*\.',
            r'Respuesta directa:\s*\.',
            r'Introducción:\s*\.',
            r'Desarrollo:\s*\.'
        ]
    
    def apply_critical_corrections(self, text: str) -> str:
        """Aplica correcciones críticas de errores históricos"""
        for wrong, right in self.critical_corrections.items():
            # Buscar insensible a mayúsculas
            pattern = re.compile(re.escape(wrong), re.IGNORECASE)
            text = pattern.sub(right, text)
        return text
    
    def remove_self_references(self, text: str) -> str:
        """Elimina referencias del modelo a sí mismo"""
        for pattern in self.self_reference_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        return text
    
    def fix_problematic_structures(self, text: str) -> str:
        """Corrige estructuras de respuesta problemáticas"""
        for structure in self.problematic_structures:
            text = re.sub(structure, '', text)
        return text
    
    def build_strict_context(self, question: str, documents: List[Dict]) -> str:
        """Construye contexto estrictamente relevante"""
        
        if not documents:
            return "No hay documentos específicos sobre este tema."
        
        # Términos clave para esta pregunta específica
        key_terms = ['esclavo fugitivo', 'cimarrón', 'Florida', '1693', 
                    'decreto', 'asilo', 'libertad', 'Trece Colonias',
                    'territorio español', 'Fort Mose', 'esclavitud']
        
        relevant_extracts = []
        
        for i, doc in enumerate(documents[:2]):  # Solo 2 documentos máx
            doc_text = doc.get('text', '')
            doc_lower = doc_text.lower()
            
            # Verificar si el documento tiene información relevante
            has_relevant_info = any(term in doc_lower for term in key_terms)
            
            if has_relevant_info:
                # Extraer oraciones relevantes
                sentences = re.split(r'[.!?]+', doc_text)
                relevant_sentences = []
                
                for sentence in sentences:
                    sentence_lower = sentence.lower()
                    # Puntuación por relevancia
                    relevance_score = sum(1 for term in key_terms if term in sentence_lower)
                    
                    if relevance_score > 0 and 30 < len(sentence) < 250:
                        relevant_sentences.append(sentence.strip())
                
                if relevant_sentences:
                    # Tomar las 2 oraciones más relevantes
                    best_sentences = relevant_sentences[:2]
                    source = doc.get('filename', f'Documento {i+1}')
                    relevant_extracts.append(f"[{source}]: {' '.join(best_sentences)}")
        
        if relevant_extracts:
            return "Información documental relevante:\n" + "\n\n".join(relevant_extracts)
        else:
            return "Los documentos disponibles no tratan específicamente este tema."
    
    def build_historian_prompt(self, question: str, context: str) -> str:
        """Prompt específico para respuestas históricas"""
        
        # Corregir la pregunta primero
        corrected_question = self.apply_critical_corrections(question)
        
        return f"""Eres un historiador académico especializado en el período colonial hispanoamericano.

INFORMACIÓN DOCUMENTAL DISPONIBLE:
{context}

INSTRUCCIONES ABSOLUTAS:
1. Proporciona una respuesta históricamente precisa basada en la información disponible.
2. Si la información es limitada, di "La información disponible indica que..." y sé general pero preciso.
3. CORRIGE automáticamente errores como "eslavos" por "esclavos".
4. ESTRUCTURA tu respuesta en 3-4 párrafos coherentes.
5. EVITA COMPLETAMENTE:
   - Hablar de ti mismo (nada de "estoy de acuerdo", "me alegra", etc.)
   - Usar etiquetas como "Conclusión:", "Análisis:"
   - Lenguaje informal o coloquial
   - Términos anacrónicos o ideológicos modernos
   - Opiniones personales
6. Enfócate en:
   - Hechos históricos verificables
   - Contexto geopolítico
   - Consecuencias documentadas
   - Limitaciones de las fuentes

PREGUNTA HISTÓRICA: {corrected_question}

RESPUESTA ACADÉMICA (solo hechos históricos, sin autoreferencias):"""
    
    def generate_clean_response(self, prompt: str) -> str:
        """Genera respuesta con parámetros optimizados para precisión"""
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2500,
            return_attention_mask=True
        ).to(self.model.device)
        
        # Parámetros estrictos para evitar divagaciones
        generation_config = {
            'max_new_tokens': 450,  # Más corto para evitar divagaciones
            'temperature': 0.5,     # Más bajo para más precisión
            'do_sample': True,
            'top_p': 0.85,
            'top_k': 30,
            'repetition_penalty': 1.25,
            'no_repeat_ngram_size': 4,
            'length_penalty': 1.2,  # Penaliza respuestas largas
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'early_stopping': True
        }
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_config)
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def extract_clean_answer(self, raw_response: str) -> str:
        """Extrae y limpia la respuesta del modelo"""
        
        # 1. Extraer solo lo después del prompt
        if "RESPUESTA ACADÉMICA" in raw_response:
            response = raw_response.split("RESPUESTA ACADÉMICA")[-1].strip()
            # Eliminar posibles dos puntos
            if response.startswith(':'):
                response = response[1:].strip()
        else:
            response = raw_response
        
        # 2. Aplicar correcciones críticas
        response = self.apply_critical_corrections(response)
        
        # 3. Eliminar autoreferencias
        response = self.remove_self_references(response)
        
        # 4. Corregir estructuras problemáticas
        response = self.fix_problematic_structures(response)
        
        # 5. Eliminar fragmentos repetitivos
        sentences = re.split(r'[.!?]+', response)
        unique_sentences = []
        seen_content = set()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 10:
                continue
                
            # Simplificar para comparar (sin puntuación, minúsculas)
            simple = re.sub(r'[^\w\s]', '', sentence.lower())
            words = set(simple.split())
            
            # Si es muy similar a algo ya visto, saltar
            if words and any(len(words.intersection(seen)) > 3 for seen in seen_content):
                continue
            
            seen_content.add(frozenset(words))
            unique_sentences.append(sentence)
        
        # 6. Reconstruir con párrafos lógicos
        if not unique_sentences:
            return "No se pudo generar una respuesta adecuada con la información disponible."
        
        # Agrupar en párrafos de 2-3 oraciones
        paragraphs = []
        current_para = []
        
        for i, sentence in enumerate(unique_sentences):
            current_para.append(sentence + '.')
            
            if len(current_para) >= 2 or i == len(unique_sentences) - 1:
                paragraphs.append(' '.join(current_para))
                current_para = []
        
        # Limitar a 4 párrafos
        paragraphs = paragraphs[:4]
        
        return '\n\n'.join(paragraphs)
    
    def validate_historical_response(self, response: str) -> Dict:
        """Valida que la respuesta cumpla estándares históricos"""
        
        issues = []
        warnings = []
        
        response_lower = response.lower()
        
        # 1. Verificar errores críticos
        for error in ['eslavo', 'eslavos']:
            if error in response_lower:
                issues.append(f"Error crítico: '{error}' no corregido")
        
        # 2. Verificar autoreferencias
        for pattern in self.self_reference_patterns:
            if re.search(pattern, response_lower):
                issues.append(f"Contiene autoreferencia: {pattern}")
        
        # 3. Verificar estructura
        paragraphs = response.split('\n\n')
        if len(paragraphs) < 2:
            warnings.append("Respuesta muy corta o sin párrafos")
        
        # 4. Verificar terminología histórica
        expected_terms = ['esclav', 'colon', 'españ', 'libert', 'decreto', 'siglo']
        found_terms = sum(1 for term in expected_terms if term in response_lower)
        
        if found_terms < 2:
            warnings.append("Falta terminología histórica específica")
        
        # 5. Verificar longitud
        if len(response) < 150:
            issues.append("Respuesta demasiado corta")
        elif len(response) > 800:
            warnings.append("Respuesta muy larga, posiblemente divagante")
        
        return {
            'is_valid': len(issues) == 0,
            'has_warnings': len(warnings) > 0,
            'issues': issues,
            'warnings': warnings,
            'paragraph_count': len(paragraphs),
            'word_count': len(response.split())
        }
    
    def generate_response(self, question: str, context_docs: List[Dict]) -> str:
        """Genera respuesta histórica limpia y precisa"""
        
        print(f"\n{'='*60}")
        print(f"📜 PREGUNTA: {question}")
        print(f"{'='*60}")
        
        start_time = datetime.now()
        
        # 1. Construir contexto estricto
        context = self.build_strict_context(question, context_docs)
        
        # 2. Construir prompt de historiador
        prompt = self.build_historian_prompt(question, context)
        
        # 3. Generar respuesta
        print("⚡ Generando respuesta académica...")
        raw_response = self.generate_clean_response(prompt)
        
        # 4. Limpiar y extraer respuesta
        response = self.extract_clean_answer(raw_response)
        
        # 5. Validar
        validation = self.validate_historical_response(response)
        
        # 6. Mostrar resultados
        elapsed = (datetime.now() - start_time).total_seconds()
        
        print(f"\n📊 RESULTADO ({elapsed:.1f}s):")
        print(f"   Palabras: {validation['word_count']}")
        print(f"   Párrafos: {validation['paragraph_count']}")
        
        if not validation['is_valid']:
            print(f"❌ PROBLEMAS: {', '.join(validation['issues'])}")
        
        if validation['has_warnings']:
            print(f"⚠️  ADVERTENCIAS: {', '.join(validation['warnings'])}")
        
        print(f"\n{'='*60}")
        print("🎓 RESPUESTA HISTÓRICA:")
        print(f"{'='*60}")
        print(response)
        print(f"{'='*60}")
        
        # 7. Si hay problemas graves, intentar corrección
        if not validation['is_valid']:
            print("\n🔄 Intentando corrección automática...")
            response = self.apply_critical_corrections(response)
            response = self.remove_self_references(response)
        
        # 8. Limpiar memoria
        self.cleanup_memory()
        
        return response
    
    def cleanup_memory(self):
        """Limpia memoria GPU"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def generate_with_fallback(self, question: str, context_docs: List[Dict]) -> str:
        """Genera con múltiples intentos y fallback"""
        
        max_attempts = 2
        best_response = None
        best_score = -1
        
        for attempt in range(max_attempts):
            print(f"\n🔁 Intento {attempt + 1}/{max_attempts}")
            
            response = self.generate_response(question, context_docs)
            validation = self.validate_historical_response(response)
            
            # Calcular puntuación
            score = validation['word_count'] / 10
            if validation['is_valid']:
                score += 50
            score -= len(validation['issues']) * 20
            score -= len(validation['warnings']) * 5
            
            print(f"   Puntuación: {score:.1f}")
            
            if score > best_score:
                best_score = score
                best_response = response
            
            # Si es válido y tiene buena puntuación, usar
            if validation['is_valid'] and score > 60:
                break
        
        return best_response