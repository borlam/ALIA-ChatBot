# -*- coding: utf-8 -*-
"""Motor de generación de respuestas con contexto"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from .config import *

class ChatEngine:
    def __init__(self):
        print("🧠 Cargando modelo salamandra-2b...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("🔤 Cargando embeddings...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        if torch.cuda.is_available():
            self.embedder = self.embedder.to(torch.device("cuda"))

    def generate_response(self, question: str, context_docs: List[Dict], max_chars: int = 2500) -> str:
        """Genera respuesta con parámetros optimizados para respuestas largas"""
        context_text = ""
        pdf_sources = {}

        # Limitar contexto para no sobrecargar el prompt
        for i, doc in enumerate(context_docs[:2]):  # Solo 2 documentos máximo
            source = doc.get('pdf_title', 'Documento PDF')
            if source not in pdf_sources:
                pdf_sources[source] = []
            pdf_sources[source].append(i + 1)

            # Tomar solo los primeros 400 caracteres de cada documento
            context_text += f"\n\n[📚 Referencia {i+1} - {source}]:\n{doc['text'][:400]}..."

        # Prompt más directivo
        prompt = f"""### INSTRUCCIÓN:
Eres un historiador experto en cultura hispánica. Responde de forma EXTENSA, DETALLADA y COMPLETA.
Usa al menos {max_chars//10} líneas en tu respuesta. NO TE CORTES a mitad de frase.

### CONTEXTO DOCUMENTAL:
{context_text}

### PREGUNTA DEL USUARIO:
{question}

### RESPUESTA EXTENSA Y DETALLADA (mínimo {max_chars//10} líneas):
"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024  # Permitir más contexto
        ).to(self.model.device)

        with torch.no_grad():
            # PARÁMETROS OPTIMIZADOS PARA RESPUESTAS LARGAS
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                min_new_tokens=MIN_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                top_p=TOP_P,
                top_k=TOP_K,
                repetition_penalty=1.15,
                no_repeat_ngram_size=4,
                length_penalty=0.8,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1,
                use_cache=True
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extraer respuesta más agresivamente
        if "RESPUESTA EXTENSA Y DETALLADA" in response:
            response = response.split("RESPUESTA EXTENSA Y DETALLADA")[-1].strip()
        elif "### RESPUESTA" in response:
            response = response.split("### RESPUESTA")[-1].strip()

        # Limpiar marcadores residuales
        for marker in ["PREGUNTA DEL USUARIO:", "### PREGUNTA", "CONTEXTO DOCUMENTAL:", "###"]:
            if marker in response:
                response = response.split(marker)[0].strip()

        # Asegurar longitud mínima
        if len(response) < max_chars // 2:
            print(f"⚠️  Respuesta muy corta ({len(response)} chars). Reintentando con más tokens...")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1200,
                    temperature=0.85,
                    do_sample=True,
                    top_p=0.97,
                    repetition_penalty=1.1,
                    length_penalty=0.5,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            if "RESPUESTA" in response:
                response = response.split("RESPUESTA")[-1].strip()

        # Limitar tamaño final
        if len(response) > max_chars:
            if "." in response[max_chars-100:max_chars+100]:
                last_period = response[:max_chars+100].rfind(".")
                response = response[:last_period+1] + ".."
            else:
                response = response[:max_chars] + "..."

        # Añadir fuentes si hay
        if pdf_sources:
            sources_text = ", ".join([f"{source}" for source in pdf_sources.keys()])
            response += f"\n\n---\n📚 **Fuentes consultadas:** {sources_text}"

        print(f"📝 Longitud respuesta final: {len(response)} caracteres")
        return response
