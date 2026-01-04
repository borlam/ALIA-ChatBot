# -*- coding: utf-8 -*-
"""Motor de chat MEJORADO - Versión 2 con filtros y validación"""

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
        
        # Diccionario de correcciones más completo
        self.common_corrections = {
            "treces colonias": "trece colonias",
            "treces colinias": "trece colonias", 
            "trece colonías": "trece colonias",
            "eslavos": "esclavos",
            "hispanoamericanavirreinal": "hispanoamericana virreinal",
            "creole": "criolla",
            "norteamérica": "Norteamérica",
            "florida": "Florida",
            "luisiana": "Luisiana",
            "moctezuma": "Moctezuma",
            "cortés": "Hernán Cortés",
            "pizarro": "Francisco Pizarro",
            "magallanes": "Magallanes"
        }
        
        # Términos históricos clave para validación
        self.historical_validation_terms = {
            'esclavos fugitivos': ['cimarrones', 'Fort Mose', '1738', 'Florida', 'libertos'],
            'trece colonias': ['británicas', 'inglesas', 'EE.UU.', 'Estados Unidos', 'colonial'],
            'españa': ['decreto 1693', 'catolicismo', 'conversión', 'asilo', 'territorios españoles']
        }
    
    def correct_question(self, question: str) -> str:
        """Corrige automáticamente errores comunes"""
        corrected = question
        
        for wrong, right in self.common_corrections.items():
            # Buscar con regex insensible a mayúsculas/minúsculas
            pattern = re.compile(re.escape(wrong), re.IGNORECASE)
            corrected = pattern.sub(right, corrected)
        
        return corrected
    
    def filter_context_documents(self, question: str, documents: List[Dict]) -> List[Dict]:
        """Filtra documentos para incluir solo los relevantes"""
        if not documents:
            return []
        
        question_lower = question.lower()
        relevant_docs = []
        
        # Palabras clave específicas para esta pregunta
        key_terms = ['esclavo', 'esclavitud', 'fugitivo', 'cimarrón', 
                    'libertad', 'españa', 'florida', 'decreto', '1693',
                    'fort mose', 'trece colonias', 'británico', 'asilo']
        
        for doc in documents:
            doc_text_lower = doc.get('text', '').lower()
            doc_metadata = doc.get('enriched_metadata', {})
            doc_themes = doc_metadata.get('themes', [])
            
            # Calcular puntuación de relevancia
            score = 0
            
            # 1. Coincidencia directa con términos clave
            for term in key_terms:
                if term in doc_text_lower:
                    score += 3
            
            # 2. Coincidencia con temas del documento
            for theme in doc_themes:
                theme_lower = theme.lower()
                for term in key_terms:
                    if term in theme_lower:
                        score += 2
            
            # 3. Coincidencia con la pregunta
            question_words = set(question_lower.split())
            doc_words = set(doc_text_lower.split())
            common_words = question_words.intersection(doc_words)
            if len(common_words) > 2:
                score += len(common_words)
            
            # Solo incluir si tiene puntuación suficiente
            if score >= 3:
                relevant_docs.append({
                    'doc': doc,
                    'score': score,
                    'reasons': f"Coincidencias: {score} puntos"
                })
        
        # Ordenar por relevancia
        relevant_docs.sort(key=lambda x: x['score'], reverse=True)
        
        # Devolver solo los documentos (sin metadatos de scoring)
        return [item['doc'] for item in relevant_docs[:3]]  # Máximo 3 documentos
    
    def build_clean_context(self, question: str, documents: List[Dict]) -> str:
        """Construye contexto limpio y relevante"""
        
        filtered_docs = self.filter_context_documents(question, documents)
        
        if not filtered_docs:
            return "No se encontraron documentos específicamente relevantes para esta pregunta."
        
        context_parts = ["### INFORMACIÓN DOCUMENTAL RELEVANTE:"]
        
        for i, doc in enumerate(filtered_docs):
            source = doc.get('pdf_title', doc.get('filename', f'Documento {i+1}'))
            
            # Extraer el fragmento más relevante
            relevant_text = self.extract_relevant_snippet(doc.get('text', ''), question)
            
            context_parts.append(f"\n📄 **{source}**")
            context_parts.append(f"*Extracto:* {relevant_text}")
            
            # Añadir metadatos útiles si existen
            enriched = doc.get('enriched_metadata', {})
            if enriched.get('key_dates'):
                dates = ', '.join(enriched['key_dates'][:2])
                context_parts.append(f"*Fechas relevantes:* {dates}")
            
            if i < len(filtered_docs) - 1:
                context_parts.append("---")
        
        return '\n'.join(context_parts)
    
    def extract_relevant_snippet(self, text: str, question: str) -> str:
        """Extrae el fragmento más relevante del texto"""
        # Limpiar texto
        text = re.sub(r'\[\d+\]', '', text)  # Eliminar referencias [1], [2], etc
        text = re.sub(r'\s+', ' ', text)     # Normalizar espacios
        
        sentences = re.split(r'[.!?]+', text)
        question_words = set(question.lower().split())
        
        best_sentence = ""
        best_score = 0
        
        for sentence in sentences:
            if 30 < len(sentence) < 300:  # Frases de longitud razonable
                sentence_lower = sentence.lower()
                sentence_words = set(sentence_lower.split())
                
                # Puntuar por palabras comunes con la pregunta
                common_words = question_words.intersection(sentence_words)
                score = len(common_words)
                
                # Bonus por términos históricos importantes
                historical_terms = ['esclavo', 'libertad', 'españa', 'florida', 'decreto', 
                                   'colonia', 'fugitivo', 'cimarrón', 'asilo', '1693']
                score += sum(1 for term in historical_terms if term in sentence_lower)
                
                if score > best_score:
                    best_score = score
                    best_sentence = sentence.strip()
        
        if best_sentence:
            # Limitar longitud y asegurar puntuación final
            snippet = best_sentence[:200]
            if len(best_sentence) > 200:
                snippet += "..."
            if snippet and snippet[-1] not in '.!?':
                snippet += "."
            return snippet
        
        # Fallback: primeras frases coherentes
        for sentence in sentences:
            if len(sentence) > 50:
                return sentence[:200] + ("..." if len(sentence) > 200 else "")
        
        return text[:200] + "..." if len(text) > 200 else text
    
    def build_prompt(self, question: str, context: str) -> str:
        """Construye prompt optimizado para calidad histórica"""
        
        corrected_question = self.correct_question(question)
        
        return f"""Eres un historiador especializado en el período colonial hispanoamericano.

<CONTEXTO_PROPORCIONADO>
{context}
</CONTEXTO_PROPORCIONADO>

<INSTRUCCIONES_ESTRICTAS>
1. Responde ÚNICAMENTE en español, con claridad y precisión histórica.
2. Usa información específica del contexto cuando sea aplicable.
3. NO inventes fechas, nombres o eventos que no estén en el contexto.
4. Si el contexto no proporciona información suficiente, di "Según el contexto disponible..." y responde de forma general.
5. Evita completamente:
   - Números entre corchetes como [1], [2], etc.
   - Listas de referencias al final
   - Texto entre paréntesis con números
   - La palabra "Fuente:" seguida de números
6. ESTRUCTURA tu respuesta:
   - Párrafo 1: Respuesta directa a la pregunta
   - Párrafo 2: Contexto histórico específico
   - Párrafo 3: Ejemplos o casos relevantes
   - Párrafo 4: Conclusión o significado histórico
7. Sé conciso (3-4 párrafos máximo).
8. Corrige sutilmente cualquier error en la pregunta original.
</INSTRUCCIONES_ESTRICTAS>

<FORMATO_PROHIBIDO>
NO uses:
- [cualquier número entre corchetes]
- "Fuente: [números]"
- Listas al final
- Secciones con títulos como "Conclusión:"
- Numeración de párrafos
</FORMATO_PROHIBIDO>

PREGUNTA: {corrected_question}

RESPUESTA DEL HISTORIADOR:"""
    
    def clean_generated_response(self, raw_response: str) -> str:
        """Limpia exhaustivamente la respuesta generada"""
        
        # 1. Extraer solo lo después de "RESPUESTA DEL HISTORIADOR:"
        if "RESPUESTA DEL HISTORIADOR:" in raw_response:
            response = raw_response.split("RESPUESTA DEL HISTORIADOR:")[-1].strip()
        else:
            response = raw_response
        
        # 2. Eliminar patrones problemáticos
        patterns_to_remove = [
            r'\[\d+\]',  # [1], [2], etc.
            r'Fuente:.*?\d',  # "Fuente: 6" o similar
            r'\(\d+\)',  # (1), (2), etc.
            r'Nota:.*',  # Notas al final
            r'Referencias:.*',  # Referencias
            r'\d+\.\s*$',  # Números al final de línea
            r'[\x00-\x1F\x7F-\x9F]',  # Caracteres de control
        ]
        
        for pattern in patterns_to_remove:
            response = re.sub(pattern, '', response)
        
        # 3. Eliminar líneas que sean solo números
        lines = response.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # No incluir líneas que sean solo números o muy cortas sin sentido
            if not stripped.isdigit() and len(stripped) > 10:
                # Limpiar números al final de la línea
                if stripped and stripped[-1].isdigit() and stripped[-2] == ' ':
                    stripped = stripped[:-2].strip()
                cleaned_lines.append(stripped)
        
        response = '\n'.join(cleaned_lines)
        
        # 4. Unificar espacios y saltos de línea
        response = re.sub(r'\s+', ' ', response)
        response = re.sub(r'\n\s*\n', '\n\n', response)
        
        # 5. Asegurar puntuación final
        if response and response[-1] not in '.!?':
            response = response + '.'
        
        # 6. Dividir en párrafos lógicos
        sentences = re.split(r'(?<=[.!?])\s+', response)
        paragraphs = []
        current_paragraph = []
        char_count = 0
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            if char_count + len(sentence) < 400 and len(current_paragraph) < 4:
                current_paragraph.append(sentence)
                char_count += len(sentence)
            else:
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                current_paragraph = [sentence]
                char_count = len(sentence)
        
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        # Limitar a 4 párrafos máximo
        paragraphs = paragraphs[:4]
        
        # 7. Aplicar correcciones finales
        final_response = '\n\n'.join(paragraphs)
        
        for wrong, right in self.common_corrections.items():
            if wrong in final_response.lower():
                pattern = re.compile(re.escape(wrong), re.IGNORECASE)
                final_response = pattern.sub(right, final_response)
        
        return final_response.strip()
    
    def validate_response(self, response: str, question: str) -> Dict:
        """Valida la calidad de la respuesta"""
        
        issues = []
        
        # 1. Verificar longitud mínima
        if len(response) < 200:
            issues.append("Respuesta demasiado corta")
        
        # 2. Verificar patrones prohibidos
        prohibited_patterns = [
            (r'\[\d+\]', "Contiene referencias entre corchetes"),
            (r'Fuente:\s*\d', "Menciona 'Fuente:' con números"),
            (r'\d+\.\s*$', "Termina con números"),
        ]
        
        for pattern, message in prohibited_patterns:
            if re.search(pattern, response):
                issues.append(message)
        
        # 3. Verificar estructura de párrafos
        paragraphs = response.split('\n\n')
        if len(paragraphs) < 2:
            issues.append("Falta estructura en párrafos")
        
        # 4. Verificar términos históricos relevantes
        question_lower = question.lower()
        if 'esclavo' in question_lower:
            expected_terms = ['esclavitud', 'libertad', 'fugitivo', 'españa', 'florida']
            found_terms = sum(1 for term in expected_terms if term in response.lower())
            if found_terms < 2:
                issues.append(f"Falta terminología histórica específica (encontrados: {found_terms})")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'paragraph_count': len(paragraphs),
            'length': len(response)
        }
    
    def generate_response(self, question: str, context_docs: List[Dict]) -> str:
        """Genera respuesta validada y limpia"""
        
        print(f"\n{'='*60}")
        print(f"🤔 PREGUNTA: {question}")
        print(f"{'='*60}")
        
        start_time = datetime.now()
        
        # 1. Construir contexto limpio
        context = self.build_clean_context(question, context_docs)
        print(f"\n📚 Contexto construido ({len(context.split())} palabras)")
        
        # 2. Construir prompt
        prompt = self.build_prompt(question, context)
        
        # 3. Tokenización
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=3000,
            padding=True
        ).to(self.model.device)
        
        # 4. Generación con parámetros ajustados
        print("⚡ Generando respuesta...")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=700,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=True
            )
        
        # 5. Decodificar
        raw_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 6. Limpiar respuesta
        response = self.clean_generated_response(raw_response)
        
        # 7. Validar
        validation = self.validate_response(response, question)
        
        # 8. Estadísticas
        elapsed = (datetime.now() - start_time).total_seconds()
        
        print(f"\n✅ RESPUESTA GENERADA ({elapsed:.1f}s):")
        print(f"📊 Longitud: {len(response)} caracteres, {len(response.split())} palabras")
        print(f"📈 Párrafos: {validation['paragraph_count']}")
        
        if not validation['valid']:
            print(f"⚠️  Advertencias: {', '.join(validation['issues'])}")
        
        print(f"\n{'='*60}")
        print("💬 RESPUESTA FINAL:")
        print(f"{'='*60}")
        print(response)
        print(f"{'='*60}")
        
        # 9. Limpiar memoria
        self.cleanup_memory()
        
        return response
    
    def cleanup_memory(self):
        """Limpia memoria GPU"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def generate_multiple_options(self, question: str, context_docs: List[Dict], n: int = 2) -> List[Dict]:
        """Genera múltiples opciones de respuesta"""
        options = []
        
        for i in range(n):
            print(f"\n🔄 Generando opción {i+1}/{n}...")
            
            # Variar ligeramente los parámetros para diversidad
            with torch.no_grad():
                inputs = self.tokenizer(
                    self.build_prompt(question, self.build_clean_context(question, context_docs)),
                    return_tensors="pt",
                    truncation=True,
                    max_length=3000
                ).to(self.model.device)
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=600,
                    temperature=0.8 if i > 0 else 0.7,  # Más diversidad en opciones posteriores
                    do_sample=True,
                    top_p=0.92,
                    top_k=50,
                    repetition_penalty=1.15,
                    num_return_sequences=1
                )
            
            raw_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            cleaned_response = self.clean_generated_response(raw_response)
            
            options.append({
                'option': i + 1,
                'response': cleaned_response,
                'length': len(cleaned_response),
                'validation': self.validate_response(cleaned_response, question)
            })
        
        return options