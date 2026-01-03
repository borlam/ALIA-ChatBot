# -*- coding: utf-8 -*-
"""Motor que lee documentos COMPLETOS y combina con conocimiento general"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import re
import hashlib
from datetime import datetime
from .config import *

class DocumentCache:
    """Cache para documentos procesados"""
    def __init__(self):
        self.cache = {}
        self.hits = 0
        self.misses = 0
    
    def get_key(self, doc: Dict) -> str:
        """Genera clave única para un documento"""
        content = doc.get('text', '')[:1000] + doc.get('pdf_title', '')
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, doc: Dict) -> Tuple[bool, str]:
        """Obtiene análisis del cache"""
        key = self.get_key(doc)
        if key in self.cache:
            self.hits += 1
            return True, self.cache[key]
        self.misses += 1
        return False, ""
    
    def set(self, doc: Dict, analysis: str):
        """Guarda análisis en cache"""
        key = self.get_key(doc)
        self.cache[key] = analysis
    
    def stats(self):
        """Estadísticas del cache"""
        return f"Cache: {self.hits} hits, {self.misses} misses, {len(self.cache)} documentos"

class ChatEngine:
    def __init__(self):
        print("🧠 Cargando modelo salamandra-7b (lectura completa de documentos)...")
        
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

        print("🔤 Cargando embeddings...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        if torch.cuda.is_available():
            self.embedder = self.embedder.to(torch.device("cuda"))
        
        # Cache para análisis de documentos
        self.doc_cache = DocumentCache()
        
        print(f"✅ Modelo listo para análisis completo de documentos")

    def analyze_complete_document(self, doc: Dict, question_keywords: List[str]) -> Dict:
        """Analiza un documento COMPLETO para extraer información relevante"""
        
        # Verificar cache primero
        cached, cached_analysis = self.doc_cache.get(doc)
        if cached:
            return eval(cached_analysis)  # Convertir string a dict
        
        text = doc['text']
        source = doc.get('pdf_title', 'Documento')
        
        analysis = {
            'source': source,
            'total_length': len(text),
            'relevant_sections': [],
            'direct_matches': [],
            'related_concepts': [],
            'key_themes': [],
            'summary': "",
            'relevance_score': 0
        }
        
        # 1. BUSCAR COINCIDENCIAS DIRECTAS EN TODO EL TEXTO
        paragraphs = text.split('\n\n')
        for para in paragraphs:
            if len(para.strip()) < 50:
                continue
                
            para_lower = para.lower()
            
            # Contar coincidencias con keywords
            matches = sum(1 for kw in question_keywords if kw in para_lower)
            
            if matches >= 1:
                # Extraer oraciones relevantes de este párrafo
                sentences = para.split('. ')
                relevant_sentences = []
                
                for sentence in sentences:
                    sentence_lower = sentence.lower()
                    sentence_matches = sum(1 for kw in question_keywords if kw in sentence_lower)
                    
                    if sentence_matches >= 1 and len(sentence) > 20:
                        relevant_sentences.append(sentence.strip())
                
                if relevant_sentences:
                    section_text = '. '.join(relevant_sentences[:3]) + '.'
                    analysis['relevant_sections'].append({
                        'text': section_text,
                        'match_count': matches,
                        'position': text.find(para) / len(text)  # Posición relativa
                    })
        
        # 2. IDENTIFICAR TEMAS CLAVE DEL DOCUMENTO
        # Analizar primeras y últimas partes para entender el tema general
        if len(text) > 1000:
            first_part = text[:500]
            last_part = text[-500:] if len(text) > 1000 else ""
            
            # Extraer palabras frecuentes
            words = re.findall(r'\b[a-záéíóúñ]{4,}\b', text.lower())
            from collections import Counter
            word_freq = Counter(words)
            common_words = [word for word, count in word_freq.most_common(10) 
                          if count > 2 and len(word) > 3]
            
            analysis['key_themes'] = common_words[:5]
        
        # 3. CREAR RESUMEN INTELIGENTE
        if len(text) > 500:
            # Tomar primeras oraciones + oraciones con keywords
            sentences = text.split('. ')
            summary_sentences = []
            
            # Primeras 2-3 oraciones
            summary_sentences.extend(sentences[:3])
            
            # Algunas oraciones del medio con keywords
            if len(sentences) > 10:
                middle_idx = len(sentences) // 2
                for i in range(middle_idx, min(middle_idx + 5, len(sentences))):
                    if any(kw in sentences[i].lower() for kw in question_keywords[:3]):
                        summary_sentences.append(sentences[i])
            
            # Últimas 1-2 oraciones
            if len(sentences) > 5:
                summary_sentences.extend(sentences[-2:])
            
            analysis['summary'] = '. '.join(set(summary_sentences))[:500] + '...'
        
        # 4. CALCULAR RELEVANCIA
        relevance_score = 0
        if analysis['relevant_sections']:
            # Basado en número de secciones relevantes y coincidencias
            relevance_score = min(10, len(analysis['relevant_sections']) * 2 + 
                                 sum(s['match_count'] for s in analysis['relevant_sections']))
        
        analysis['relevance_score'] = relevance_score
        
        # Guardar en cache
        self.doc_cache.set(doc, str(analysis))
        
        return analysis

    def extract_keywords(self, question: str) -> List[str]:
        """Extrae palabras clave mejoradas"""
        question_lower = question.lower()
        
        # Palabras a ignorar
        stop_words = {'que', 'de', 'la', 'el', 'los', 'las', 'un', 'una', 'unos', 'unas', 
                     'y', 'o', 'pero', 'por', 'para', 'con', 'sin', 'sobre', 'bajo',
                     'entre', 'hacia', 'desde', 'durante', 'mediante', 'según', 'ante',
                     'como', 'cuando', 'donde', 'qué', 'quién', 'cuál', 'cómo', 'por qué'}
        
        # Extraer todas las palabras
        words = re.findall(r'\b[a-záéíóúñ]+\b', question_lower)
        
        # Filtrar y priorizar
        filtered_words = []
        for word in words:
            if word not in stop_words and len(word) > 2:
                # Priorizar palabras más largas y específicas
                score = len(word)
                if word in ['historia', 'esclavos', 'colonias', 'españoles', 'libertad', 
                           'documentos', 'textos', 'siglo', 'época', 'periodo']:
                    score += 5
                filtered_words.append((word, score))
        
        # Ordenar por score
        filtered_words.sort(key=lambda x: x[1], reverse=True)
        
        return [word for word, score in filtered_words[:15]]  # Más keywords

    def generate_comprehensive_context(self, question: str, documents: List[Dict]) -> str:
        """Crea un contexto COMPLETO combinando análisis de todos los documentos"""
        
        print(f"📚 Analizando {len(documents)} documentos COMPLETOS...")
        
        question_keywords = self.extract_keywords(question)
        print(f"   KeyworFds identificadas: {', '.join(question_keywords[:5])}...")
        
        # Analizar TODOS los documentos
        all_analyses = []
        for i, doc in enumerate(documents):
            print(f"   📄 Analizando documento {i+1}/{len(documents)}...")
            analysis = self.analyze_complete_document(doc, question_keywords)
            all_analyses.append(analysis)
        
        # Ordenar por relevancia
        all_analyses.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Construir contexto detallado
        context_parts = []
        
        # 1. DOCUMENTOS CON INFORMACIÓN DIRECTA
        direct_docs = [a for a in all_analyses if a['relevance_score'] >= 3]
        if direct_docs:
            context_parts.append("### 📚 DOCUMENTOS CON INFORMACIÓN DIRECTA SOBRE LA PREGUNTA:")
            
            for analysis in direct_docs[:3]:  # Máximo 3 documentos principales
                context_parts.append(f"\n**{analysis['source']}** (relevancia: {analysis['relevance_score']}/10)")
                context_parts.append(f"*Extractos relevantes:*")
                
                for section in analysis['relevant_sections'][:2]:  # Máximo 2 secciones por doc
                    context_parts.append(f"- {section['text'][:300]}...")
                
                if analysis['summary']:
                    context_parts.append(f"*Resumen del documento:* {analysis['summary']}")
            
            context_parts.append("")
        
        # 2. DOCUMENTOS CON INFORMACIÓN RELACIONADA
        related_docs = [a for a in all_analyses if 1 <= a['relevance_score'] < 3]
        if related_docs and len(direct_docs) < 3:
            context_parts.append("### 📄 DOCUMENTOS CON INFORMACIÓN RELACIONADA:")
            
            for analysis in related_docs[:2]:
                context_parts.append(f"\n**{analysis['source']}**")
                if analysis['key_themes']:
                    context_parts.append(f"*Temas principales:* {', '.join(analysis['key_themes'])}")
                if analysis['summary']:
                    context_parts.append(f"*Contenido general:* {analysis['summary'][:200]}...")
            
            context_parts.append("")
        
        # 3. METADATA DE ANÁLISIS
        context_parts.append(f"### 📊 METADATA DEL ANÁLISIS:")
        context_parts.append(f"- Total documentos analizados: {len(documents)}")
        context_parts.append(f"- Documentos con información directa: {len(direct_docs)}")
        context_parts.append(f"- Documentos con información relacionada: {len(related_docs)}")
        context_parts.append(f"- Palabras clave buscadas: {', '.join(question_keywords[:8])}")
        context_parts.append(f"- {self.doc_cache.stats()}")
        
        return '\n'.join(context_parts)

    def generate_response(self, question: str, context_docs: List[Dict], max_chars: int = 2500) -> str:
        """Genera respuesta basada en ANÁLISIS COMPLETO de documentos + conocimiento general"""
        
        start_time = datetime.now()
        
        # 1. ANÁLISIS COMPLETO DE TODOS LOS DOCUMENTOS
        detailed_context = self.generate_comprehensive_context(question, context_docs)
        
        # 2. PROMPT PARA ANÁLISIS PROFUNDO
        prompt = f"""### ROL Y CONTEXTO:
Eres la Inteligencia Artifical regerIA, especializada en historia hispanoamericana virreinal.
Tienes acceso a tu conocimiento académico general Y a una colección de documentos históricos que has analizado meticulosamente.

### PREGUNTA DE INVESTIGACIÓN:
"{question}"

### ANÁLISIS COMPLETO DE LA COLECCIÓN DOCUMENTAL:
{detailed_context}

### METODOLOGÍA DE RESPUESTA:
Como historiador experto, SÍGUETE estos pasos:

**PASO 1: ANALIZAR LA PREGUNTA**
- Identificar el núcleo histórico de la pregunta
- Determinar período, actores y conceptos clave

**PASO 2: INTEGRAR FUENTES DOCUMENTALES**
- Usar EXTRACTOS ESPECÍFICOS de los documentos cuando sean relevantes
- Citar documentos por nombre cuando los uses
- Señalar si hay información contradictoria o complementaria entre documentos

**PASO 3: APLICAR CONOCIMIENTO ACADÉMICO GENERAL**
- Complementar con tu conocimiento histórico general
- Contextualizar dentro de marcos teóricos historiográficos
- Señalar consensos académicos cuando apliquen

**PASO 4: CONSTRUIR RESPUESTA ESTRUCTURADA**
1. **RESPUESTA DIRECTA**: Comenzar con respuesta clara y concisa
2. **EVIDENCIA DOCUMENTAL**: Presentar evidencia específica de los documentos
3. **CONTEXTO HISTÓRICO**: Ampliar con conocimiento general relevante
4. **ANÁLISIS CRÍTICO**: Ofrecer interpretación historiográfica
5. **CONCLUSIÓN**: Sintetizar los hallazgos principales

### FORMATO DE CITACIÓN:
- Para información de documentos: "[Documento: Nombre del documento] cita o información específica"
- Para conocimiento general: sin marca especial
- Ser explícito sobre el origen de cada pieza de información

### EJEMPLO DE ESTRUCTURA IDEAL:
**Respuesta directa**: [Respuesta concisa]
**Evidencia documental**: 
  - Del documento X: "cita específica"
  - Del documento Y: análisis de contenido relevante
**Contexto histórico general**: [Explicación ampliada]
**Análisis crítico**: [Interpretación historiográfica]
**Conclusión**: [Síntesis final]

### RESPUESTA DE regerIA:
"""
        
        # 3. CALCULAR LÍMITES DE TOKENS
        prompt_tokens = len(self.tokenizer.encode(prompt))
        max_total_tokens = 4096  # Límite conservador para 7B en 4-bit
        available_tokens = max_total_tokens - prompt_tokens - 200
        
        max_gen_tokens = min(1000, available_tokens)
        max_gen_tokens = max(300, max_gen_tokens)
        
        print(f"📝 Tokens: prompt={prompt_tokens}, disponibles={available_tokens}, generación={max_gen_tokens}")
        
        # 4. GENERACIÓN CON AMPLIO CONTEXTO
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_total_tokens - 300
        ).to(self.model.device)
        
        print("🤖 Generando respuesta con análisis profundo...")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_gen_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.93,
                top_k=60,
                repetition_penalty=1.15,
                no_repeat_ngram_size=4,
                length_penalty=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
        
        # 5. PROCESAR Y MEJORAR RESPUESTA
        raw_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extraer solo la parte después del marcador
        response_marker = "RESPUESTA DE regerIA:"
        if response_marker in raw_response:
            response = raw_response.split(response_marker)[-1].strip()
        else:
            # Buscar marcadores alternativos
            for marker in ["Respuesta:", "RESPUESTA:", "### RESPUESTA"]:
                if marker in raw_response:
                    parts = raw_response.split(marker)
                    if len(parts) > 1:
                        response = parts[-1].strip()
                        break
            else:
                response = raw_response[-1500:]  # Tomar último segmento
        
        # 6. POST-PROCESAMIENTO PARA CLARIDAD
        # Asegurar formato de citas
        response = re.sub(r'\[Documento:\s*(.+?)\](.+?)(?=\[|$)', 
                         r'**[Documento: \1]** \2', response)
        
        # Limpiar repeticiones
        response = re.sub(r'(\b\w+\b)(?:\s+\1\b)+', r'\1', response)
        
        # Añadir etiquetas de sección si no las tiene
        sections_needed = ['Respuesta directa', 'Evidencia documental', 
                          'Contexto histórico', 'Análisis crítico', 'Conclusión']
        
        for section in sections_needed:
            if section.lower() not in response.lower() and len(response) > 500:
                # Encontrar punto natural para insertar
                sentences = response.split('. ')
                if len(sentences) > 8:
                    insert_point = min(3, len(sentences)//3)
                    sentences.insert(insert_point, f"**{section}:**")
                    response = '. '.join(sentences)
        
        # 7. AÑADIR METADATA FINAL
        elapsed = (datetime.now() - start_time).total_seconds()
        
        metadata = f"\n\n---\n"
        metadata += f"**📊 METADATA DE LA RESPUESTA**\n"
        metadata += f"- ⏱️ Tiempo de análisis: {elapsed:.1f} segundos\n"
        metadata += f"- 📚 Documentos analizados: {len(context_docs)}\n"
        metadata += f"- 🔍 Palabras clave identificadas: {len(self.extract_keywords(question))}\n"
        metadata += f"- 📝 Longitud respuesta: {len(response)} caracteres\n"
        metadata += f"- 🧠 Modelo: Salamandra-7B + análisis completo de documentos"
        
        response += metadata
        
        # 8. LIMITAR LONGITUD
        if len(response) > max_chars:
            # Buscar último punto lógico
            if "." in response[max_chars-500:max_chars]:
                last_period = response[:max_chars].rfind(".")
                response = response[:last_period+1]
            else:
                response = response[:max_chars] + "..."
        
        print(f"✅ Respuesta generada en {elapsed:.1f}s: {len(response)} caracteres")
        print(f"   📊 Cache: {self.doc_cache.stats()}")
        
        # Limpiar memoria
        self.cleanup_memory()
        
        return response.strip()

    def cleanup_memory(self):
        """Limpia memoria GPU"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()