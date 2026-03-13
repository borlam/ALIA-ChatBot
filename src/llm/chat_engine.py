# -*- coding: utf-8 -*-
"""Motor de chat SIMPLIFICADO (usa análisis ya hecho)"""

import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict
import re
from datetime import datetime
from ..system.config import *

# Variable global para tracking
_ACTIVE_MODEL_INSTANCE = None

class ChatEngine:
    def __init__(self, model_key: str = None):
        """Inicializa el motor de chat con un modelo específico"""
        global _ACTIVE_MODEL_INSTANCE
        
        # 1. Liberar modelo anterior si existe
        if _ACTIVE_MODEL_INSTANCE is not None:
            print("🔄 Cambiando de modelo: liberando anterior...")
            try:
                _ACTIVE_MODEL_INSTANCE.unload_model()
            except:
                pass
            _ACTIVE_MODEL_INSTANCE = None
        
        # 2. DEBUG: Mostrar qué modelo vamos a cargar
        print(f"\n🧠 INICIALIZANDO CHAT ENGINE")
        print(f"   Modelo solicitado: {model_key or 'por defecto'}")
        
        # 3. Actualizar configuración si se especifica un modelo
        if model_key and model_key in get_available_models_list():
            print(f"   Cambiando modelo activo a: {model_key}")
            set_active_model(model_key)
        else:
            print(f"   Usando modelo activo actual: {ACTIVE_MODEL_KEY}")
        
        # 4. Obtener información del modelo actual DESPUÉS de actualizar
        self.model_info = get_active_model_info()
        print(f"   Configuración cargada: {self.model_info['name']}")
        
        # 5. Cargar el modelo usando la configuración actual
        self._load_model()
        
        # 6. Registrar como instancia activa
        _ACTIVE_MODEL_INSTANCE = self
        
        print(f"✅ Modelo {self.model_info['display_name']} cargado en modo optimizado")
    
    def _load_model(self):
        """Carga el modelo usando la configuración actual"""
        print(f"\n🧠 Cargando modelo {self.model_info['name']} (modo optimizado)...")
        print(f"📊 Configuración: {self.model_info['max_tokens']} tokens máx, {TEMPERATURE} temperatura")
        
        model_name = self.model_info["name"]

        # Detectar el mejor dtype según la GPU disponible
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            # bfloat16 disponible en Ampere (8.x) y superior (A100, H100, L4...)
            compute_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
            torch_dtype = compute_dtype
            print(f"   🖥️  GPU compute capability: {cap[0]}.{cap[1]} → dtype: {compute_dtype}")
        else:
            compute_dtype = torch.float32
            torch_dtype = torch.float32

        # Configurar cuantización 4-bit
        extra_bnb = {"llm_int8_enable_fp32_cpu_offload": True} if "40b" in model_name.lower() else {}
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            **extra_bnb
        )
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

        except Exception as e:
            print(f"⚠️ Error cargando modelo con cuantización: {e}")
            print("🔄 Intentando cargar sin cuantización (float16/float32)...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch_dtype,
                    trust_remote_code=True
                )
            except Exception as e2:
                print(f"⚠️ Float16 falló ({e2}), cargando en CPU float32...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="cpu",
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                )
        
        # Configurar tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model_loaded = True
    
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

        for doc in documents[:6]:
            text = doc["text"]

            if len(text) > 400:
                text = text[:400]

            parts.append(text)

        return "\n\n".join(parts)

    def build_prompt_with_confidence(self, question: str, context: str, confidence: str) -> str:
        tone = {
            "high": "Responde con seguridad y detalle basándote únicamente en los fragmentos anteriores.",
            "medium": "Responde de forma natural basándote únicamente en los fragmentos anteriores, indicando matices si es necesario.",
        }[confidence]

        context_block = f"""Fragmentos de los documentos indexados:
{context}
"""

        return f"""
Eres un asistente documental. Tu ÚNICA fuente de información son los fragmentos de documentos proporcionados a continuación.

Reglas ESTRICTAS:
- Responde EXCLUSIVAMENTE con información presente en los fragmentos proporcionados.
- NO uses conocimiento general ni información que no aparezca en los documentos.
- Si los fragmentos no contienen información suficiente, responde exactamente: "No encuentro información sobre eso en los documentos disponibles."
- No inventes, no extrapoles, no añadas contexto externo.
- {tone}
- Responde siempre en español.

{context_block}
Pregunta:
{question}

Respuesta:
"""

    def generate_response(self, question: str, context_docs: List[Dict], max_chars: int = 2000) -> str:
        """Genera respuesta RÁPIDA usando análisis pre-existente"""
        
        start_time = datetime.now()
        
        confidence = self.compute_confidence(context_docs)

        # Sin documentos relevantes: responder directamente sin llamar al modelo
        if confidence == "low" or not context_docs:
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"⚠️  Sin documentos relevantes ({elapsed:.1f}s)")
            return "No encuentro información sobre eso en los documentos disponibles."

        context = self.build_intelligent_context(question, context_docs)
        prompt = self.build_prompt_with_confidence(question, context, confidence)

        if confidence == "high":
            temperature = 0.6
        else:
            temperature = 0.7

        # Ajustar tokens según modelo
        model_name = self.model_info["name"]
        if "40b" in model_name.lower():
            max_length = 3500
            max_new_tokens = 800
        else:
            max_length = 2500
            max_new_tokens = self.model_info["max_tokens"]

        # Tokenización
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        ).to(self.model.device)
        
        # Generación
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
        
        # Procesamiento
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

    def unload_model(self):
        """Libera completamente el modelo y la memoria GPU"""
        try:
            if hasattr(self, "model"):
                del self.model
                self.model = None
            if hasattr(self, "tokenizer"):
                del self.tokenizer
                self.tokenizer = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            gc.collect()
            self.model_loaded = False
            print("🧹 Modelo descargado y memoria liberada")

        except Exception as e:
            print(f"⚠️ Error liberando memoria: {e}")