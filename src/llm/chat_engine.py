# -*- coding: utf-8 -*-
"""Motor de chat SIMPLIFICADO (usa análisis ya hecho)"""

import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict
import re
from datetime import datetime
from ..system.config import *
import os 

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
        
        # DETECTAR si es GGUF
        if "gguf" in model_name.lower():
            print("🔧 Detectado modelo GGUF, cargando con llama-cpp...")
            self._load_gguf_model()
            return  # IMPORTANTE: Salir de la función aquí
        
        # ===== SOLO PARA MODELOS TRANSFORMERS (NO GGUF) =====
        
        # Configurar cuantización según modelo
        if "40b" in model_name.lower():
            # Para ALIA-40B original (si la mantienes)
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
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
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
        except Exception as e:
            print(f"⚠️ Error cargando modelo {model_name}: {e}")
            print("🔄 Intentando cargar sin cuantización...")
            
            # Fallback: cargar sin cuantización
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True
            )
        
        # Configurar tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model_type = "transformers"  # <-- IMPORTANTE
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
        model_name = self.model_info["name"]
        if "40b" in model_name.lower():
            max_length = 3500
            max_new_tokens = 800
        else:
            max_length = 2500
            max_new_tokens = self.model_info["max_tokens"]

        # DETECTAR TIPO DE MODELO Y GENERAR SEGÚN CORRESPONDA
        if hasattr(self, 'model_type') and self.model_type == "gguf":
            # GENERACIÓN PARA MODELO GGUF
            response = self.model(
                prompt,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=TOP_P,
                top_k=TOP_K,
                stop=["</s>", "###", "\n\n"],
                echo=False
            )
            
            # Extraer texto de la respuesta GGUF
            if isinstance(response, dict) and "choices" in response:
                response = response["choices"][0]["text"].strip()
            else:
                response = str(response).strip()
                
        else:
            # GENERACIÓN PARA MODELO TRANSFORMERS (TU CÓDIGO ORIGINAL)
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
        
        # Extraer solo la respuesta (común para ambos tipos)
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


    def _load_gguf_model(self):
        """Carga modelo GGUF con llama-cpp"""
        from llama_cpp import Llama
        from huggingface_hub import hf_hub_download
        
        # Crear directorio para modelos si no existe
        os.makedirs("models", exist_ok=True)
        
        # Descargar modelo
        model_path = hf_hub_download(
            repo_id=self.model_info["name"],
            filename="ALIA-40b.Q8_0.gguf",
            cache_dir="models"
        )
        
        # Cargar con llama-cpp
        self.model = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_gpu_layers=-1,  # Todas las capas en GPU
            n_batch=512,
            verbose=False
        )
        
        self.model_type = "gguf"  # <-- IMPORTANTE: Establecer el tipo
        self.model_loaded = True
        print("✅ Modelo GGUF cargado")