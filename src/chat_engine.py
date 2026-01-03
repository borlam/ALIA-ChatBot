# -*- coding: utf-8 -*-
"""Motor optimizado para precisión con Salamandra-7B"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import re
from .config import *

class ChatEngine:
    def __init__(self):
        print("🧠 Cargando modelo salamandra-7b (optimizado para precisión)...")
        
        # CONFIGURACIÓN DE CUANTIZACIÓN 4-BIT
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
            trust_remote_code=False
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("🔤 Cargando embeddings para precisión...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        if torch.cuda.is_available():
            self.embedder = self.embedder.to(torch.device("cuda"))
        
        print(f"✅ Modelo cargado en 4-bit")
        self.print_memory_usage()

    def print_memory_usage(self):
        """Imprime uso de memoria"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"📊 Memoria GPU: {allocated:.1f}GB / {reserved:.1f}GB / {total:.1f}GB")

    def extract_key_information(self, documents: List[Dict]) -> str:
        """Extrae y resume información clave de múltiples documentos"""
        key_info = {}
        
        for i, doc in enumerate(documents):
            source = doc.get('pdf_title', f'Documento {i+1}')
            text = doc['text']
            
            # Extraer oraciones más relevantes (primeras y con palabras clave)
            sentences = text.split('. ')
            if len(sentences) > 3:
                # Tomar primeras oraciones + algunas del medio
                selected = sentences[:2] + sentences[len(sentences)//2:len(sentences)//2+1]
                summary = '. '.join(selected) + '.'
            else:
                summary = text[:300] + '...' if len(text) > 300 else text
            
            key_info[source] = summary
        
        # Formatear información clave
        formatted = ""
        for source, info in key_info.items():
            formatted += f"\n\n📄 **{source}**:\n{info}"
        
        return formatted

    def generate_response(self, question: str, context_docs: List[Dict], max_chars: int = 2000) -> str:
        """Genera respuesta PRECISA usando TODOS los documentos relevantes"""
        
        print(f"🔍 Analizando {len(context_docs)} documentos para: '{question[:80]}...'")
        
        # 1. EXTRAER INFORMACIÓN CLAVE DE TODOS LOS DOCUMENTOS
        context_text = self.extract_key_information(context_docs)
        
        # 2. CONTAR FUENTES ÚNICAS
        pdf_sources = {}
        for i, doc in enumerate(context_docs):
            source = doc.get('pdf_title', f'Documento {i+1}')
            if source not in pdf_sources:
                pdf_sources[source] = 0
            pdf_sources[source] += 1
        
        # 3. PROMPT OPTIMIZADO PARA PRECISIÓN
        prompt = f"""### ROL:
Eres un historiador experto en cultura hispánica. Tu tarea es responder PRECISAMENTE basándote EXCLUSIVAMENTE en los documentos proporcionados.

### DOCUMENTOS DE REFERENCIA ({len(context_docs)} documentos):
{context_text if context_text else "No hay documentos específicos disponibles."}

### REGLAS ESTRICTAS:
1. Responde ÚNICAMENTE con información presente en los documentos anteriores
2. Si algo no está en los documentos, di "No encuentro esa información en los documentos"
3. Sé preciso y cita información específica cuando sea posible
4. No inventes nombres, fechas, eventos o referencias
5. Si los documentos son contradictorios, menciónalo

### PREGUNTA:
{question}

### PROCESO DE ANÁLISIS:
1. Identificar qué documentos contienen información relevante
2. Extraer los hechos clave
3. Sintetizar una respuesta precisa

### RESPUESTA PRECISA BASADA EN DOCUMENTOS:"""

        # 4. CALCULAR TOKENS DISPONIBLES
        # Estimación conservadora para 7B
        max_context_tokens = 2048  # Límite seguro para 7B en 4-bit
        prompt_tokens = len(self.tokenizer.encode(prompt))
        available_tokens = max_context_tokens - prompt_tokens - 100  # Margen
        
        max_gen_tokens = min(600, available_tokens)  # Máximo razonable
        max_gen_tokens = max(100, max_gen_tokens)    # Mínimo razonable
        
        print(f"📝 Tokens: prompt={prompt_tokens}, disponibles={available_tokens}, generación={max_gen_tokens}")

        # 5. TOKENIZACIÓN CON MANEJO DE TRUNCAMIENTO
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_context_tokens - 200  # Dejar espacio para respuesta
            ).to(self.model.device)
        except Exception as e:
            print(f"⚠️ Error tokenizando: {e}")
            # Versión de respaldo con menos contexto
            if len(context_text) > 3000:
                context_text = context_text[:3000] + "... [texto truncado por longitud]"
            prompt = prompt.replace(context_text, context_text)
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1900).to(self.model.device)

        # 6. GENERACIÓN CON PARÁMETROS PARA PRECISIÓN
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_gen_tokens,
                min_new_tokens=80,
                temperature=0.5,  # BAJO para máxima precisión
                do_sample=False,   # greedy decoding para más consistencia
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.25,  # Alto para evitar repeticiones
                no_repeat_ngram_size=4,
                length_penalty=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )

        # 7. DECODIFICACIÓN Y VALIDACIÓN
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extraer SOLO la parte después de "RESPUESTA PRECISA BASADA EN DOCUMENTOS:"
        response_marker = "RESPUESTA PRECISA BASADA EN DOCUMENTOS:"
        if response_marker in response:
            response = response.split(response_marker)[-1].strip()
        
        # También intentar con otras variantes
        for marker in ["Respuesta:", "respuesta:", "### RESPUESTA", "RESPUESTA:"]:
            if marker in response and response.find(marker) < 100:  # Solo si está cerca del inicio
                response = response.split(marker)[-1].strip()
        
        # 8. POST-PROCESAMIENTO PARA PRECISIÓN
        # Eliminar cualquier referencia al prompt
        for marker in ["### ROL:", "### DOCUMENTOS:", "### REGLAS:", "### PREGUNTA:", "### PROCESO:"]:
            if marker in response:
                response = response.split(marker)[0].strip()
        
        # Verificar que no se inventen URLs
        urls = re.findall(r'https?://\S+', response)
        if urls and len(context_docs) < 3:  # Si hay URLs pero pocos documentos, sospechoso
            response = re.sub(r'https?://\S+', '[referencia a documento]', response)
        
        # 9. AÑADIR METADATA DE FUENTES
        if pdf_sources and len(context_docs) > 0:
            sources_list = list(pdf_sources.keys())
            if sources_list:
                if len(sources_list) <= 3:
                    sources_text = ", ".join(sources_list)
                else:
                    sources_text = f"{sources_list[0]}, {sources_list[1]} y {len(sources_list)-2} más"
                
                response += f"\n\n---\n📚 **Documentos consultados ({len(context_docs)}):** {sources_text}"
        
        # 10. VALIDACIÓN DE CALIDAD
        # Verificar que no sea demasiado genérica
        generic_phrases = [
            "según los documentos", "basándome en la información", 
            "los documentos indican", "la información proporcionada"
        ]
        
        has_specific_info = any(phrase in response.lower() for phrase in generic_phrases)
        if not has_specific_info and len(context_docs) > 0:
            response += "\n\n💡 *Nota: Esta respuesta se basa en el análisis de los documentos proporcionados.*"
        
        # 11. LIMITAR LONGITUD FINAL
        if len(response) > max_chars:
            # Buscar el último punto completo antes del límite
            if "." in response[max_chars-300:max_chars]:
                last_period = response[:max_chars].rfind(".")
                response = response[:last_period+1]
            else:
                response = response[:max_chars] + "..."
        
        print(f"✅ Respuesta generada: {len(response)} caracteres, basada en {len(context_docs)} documentos")
        
        # 12. LIMPIEZA DE MEMORIA
        self.cleanup_memory()
        
        return response.strip()

    def cleanup_memory(self):
        """Limpia memoria GPU de forma segura"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Pequeña pausa para permitir liberación
            import time
            time.sleep(0.1)

    def validate_response(self, response: str, context_docs: List[Dict]) -> bool:
        """Valida que la respuesta sea coherente con los documentos"""
        if not response or len(response) < 20:
            return False
        
        # Verificar que mencione algún documento si hay documentos
        if context_docs and len(context_docs) > 0:
            doc_mentions = 0
            for doc in context_docs:
                title = doc.get('pdf_title', '').lower()
                if title and title in response.lower():
                    doc_mentions += 1
            
            # Si hay múltiples documentos pero no se menciona ninguno
            if len(context_docs) > 2 and doc_mentions == 0:
                print(f"⚠️  La respuesta no menciona documentos específicos")
                return False
        
        return True