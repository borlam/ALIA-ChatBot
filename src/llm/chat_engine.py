# -*- coding: utf-8 -*-
"""Motor de chat SIMPLIFICADO (usa análisis ya hecho)"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict
import re
from datetime import datetime
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

    def compute_confidence(self, documents: List[Dict]) -> str:
        if not documents:
            return "low"

        scores = [doc.get("score", 0) for doc in documents[:5]]
        avg_score = sum(scores) / len(scores)

        if avg_score >= 0.75 and len(documents) >= 3:
            return "high"
        elif avg_score >= 0.55:
            return "medium"
        else:
            return "low"

    def build_intelligent_context(self, question: str, documents: List[Dict]) -> str:
        if not documents:
            return ""

        parts = []

        for doc in documents[:6]:  # ⬅️ subir a 6
            text = doc["text"]

            if len(text) > 400:
                text = text[:400]

            parts.append(text)

        return "\n\n".join(parts)

    def build_prompt_with_confidence(self, question: str, context: str, confidence: str) -> str:
        tone = {
            "high": "Responde con seguridad y detalle.",
            "medium": "Responde de forma natural, indicando matices si es necesario.",
            "low": (
                "Responde de forma conversacional. "
                "Si no estás completamente seguro, indícalo de manera natural "
                "y evita afirmaciones categóricas."
            )
        }[confidence]

        return f"""
Eres regerIA, un asistente conversacional experto en historia hispanoamericana.
Hablas de forma clara, cercana y natural.

Contexto disponible (puede ser parcial o incompleto):
{context}

Instrucciones:
- Usa el contexto si es relevante.
- Si el contexto no es suficiente, razona con tu conocimiento general.
- No inventes datos específicos que no aparezcan en el contexto.
- {tone}
- Responde siempre en español.

Pregunta:
{question}

Respuesta:
"""


    def build_optimized_prompt(self, question: str, context: str) -> str:
        return f"""
Eres regerIA, un asistente conversacional experto en historia hispanoamericana.
Hablas de forma clara, natural y cercana.

Contexto disponible (puede ser parcial o incompleto):
{context}

Instrucciones:
- Usa el contexto si es relevante.
- Si el contexto no es suficiente, razona con tu conocimiento general.
- Si no estás completamente seguro, dilo de forma natural.
- No inventes citas específicas si no aparecen en el contexto.
- Responde siempre en español.

Pregunta:
{question}

Respuesta:
"""
   
    def generate_response(self, question: str, context_docs: List[Dict], max_chars: int = 2000) -> str:
        """Genera respuesta RÁPIDA usando análisis pre-existente"""
        
        start_time = datetime.now()
        
        confidence = self.compute_confidence(context_docs)
        context = self.build_intelligent_context(question, context_docs)
        prompt = self.build_prompt_with_confidence(question, context, confidence)

        if confidence == "high":
            temperature = 0.6
        elif confidence == "medium":
            temperature = 0.7
        else:
            temperature = 0.85

        # 3. Tokenización eficiente
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2500  # Menor límite para más velocidad
        ).to(self.model.device)
        
        # 4. Generación con parámetros optimizados
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=600,  # Suficiente para respuestas claras
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.15,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 5. Procesamiento simple
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extraer solo la respuesta
        if "RESPUESTA:" in response:
            response = response.split("RESPUESTA:")[-1].strip()
        elif "respuesta:" in response:
            response = response.split("respuesta:")[-1].strip()
        
        # Limitar longitud
        if len(response) > max_chars:
            if "." in response[max_chars-200:max_chars]:
                last_period = response[:max_chars].rfind(".")
                response = response[:last_period+1]
        
        # Estadísticas
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"✅ Respuesta en {elapsed:.1f}s, {len(response)} caracteres")
        
        # Limpiar memoria
        self.cleanup_memory()
        
        return response.strip()
    
    def cleanup_memory(self):
        """Limpia memoria GPU"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()