# -*- coding: utf-8 -*-
"""Motor de chat SIMPLIFICADO (usa análisis ya hecho)"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict
import re
from datetime import datetime
from ..system.config import *

class ChatEngine:
    def __init__(self, model_key: str = None):
        """Inicializa el motor de chat con un modelo específico"""
        
        # Usar modelo por defecto si no se especifica
        if model_key and model_key in get_available_models_list():
            set_active_model(model_key)
        
        print(f"🧠 Cargando modelo {MODEL_NAME} (modo optimizado)...")
        print(f"📊 Configuración: {MAX_TOKENS} tokens máx, {TEMPERATURE} temperatura")
        
        # Configurar cuantización según modelo
        model_info = get_active_model_info()
        
        if "40b" in model_info["name"].lower():
            # Para ALIA-40B, usar cuantización más agresiva
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                llm_int8_enable_fp32_cpu_offload=True  # Descargar a CPU si es necesario
            )
        else:
            # Para Salamandra 2B/7B
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
            
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
        except Exception as e:
            print(f"⚠️ Error cargando modelo {MODEL_NAME}: {e}")
            print("🔄 Intentando cargar sin cuantización...")
            
            # Fallback: cargar sin cuantización
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True
            )
        
        # Configurar tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"✅ Modelo {model_info['display_name']} cargado en modo optimizado")
        
        # Almacenar info del modelo
        self.model_info = model_info

    def compute_confidence(self, documents: List[Dict]) -> str:
        # ... (mantén esta función igual que antes) ...
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
        # ... (mantén esta función igual que antes) ...
        if not documents:
            return ""

        parts = []

        for doc in documents[:6]:
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

        context_block = ""
        if context.strip():
            context_block = f"""
Contexto documental (puede ser parcial, sesgado o no relevante para la pregunta):
{context}
"""

        return f"""
Eres regerIA, un asistente experto en historia hispanoamericana.

Tu función es explicar procesos históricos con rigor, claridad y sentido crítico.
No debes resumir documentos ni justificar posturas políticas o imperiales.

Instrucciones IMPORTANTES:
- Usa el contexto SOLO si aporta información directamente relevante.
- Si el contexto no es pertinente, ignóralo por completo.
- Distingue entre hechos históricos comprobados y valoraciones.
- Evita idealizar o demonizar a personas, pueblos o imperios.
- No inventes datos concretos si no estás seguro.
- {tone}
- Responde siempre en español, con un estilo claro y cercano.
{context_block}
Pregunta:
{question}

Respuesta:
"""


    def generate_response(self, question: str, context_docs: List[Dict], max_chars: int = 2000) -> str:
        """Genera respuesta RÁPIDA usando análisis pre-existente"""
        
        start_time = datetime.now()
        
        confidence = self.compute_confidence(context_docs)
        context = ""
        if confidence != "low":
            context = self.build_intelligent_context(question, context_docs)
        prompt = self.build_prompt_with_confidence(question, context, confidence)

        if confidence == "high":
            temperature = 0.6
        elif confidence == "medium":
            temperature = 0.7
        else:
            temperature = 0.85

        # Ajustar tokens según modelo
        if "40b" in self.model_info["name"].lower():
            max_length = 3500  # ALIA necesita más contexto
            max_new_tokens = 800
        else:
            max_length = 2500
            max_new_tokens = MAX_TOKENS

        # 3. Tokenización eficiente
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        ).to(self.model.device)
        
        # 4. Generación con parámetros optimizados
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=TOP_P,
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
        print(f"✅ Respuesta en {elapsed:.1f}s, {len(response)} caracteres (Modelo: {self.model_info['display_name']})")
        
        # Limpiar memoria
        self.cleanup_memory()
        
        return response.strip()
    
    def cleanup_memory(self):
        """Limpia memoria GPU"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_model_info(self):
        """Obtiene información del modelo actual"""
        return self.model_info