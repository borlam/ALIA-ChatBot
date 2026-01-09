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
        """Prompt para modelos Transformers (estilo original)"""
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

    def _build_gguf_prompt(self, question: str, context: str, confidence: str) -> str:
        """Prompt ESPECÍFICO optimizado para modelos GGUF/ALIA"""
        
        # Instrucciones claras y directas
        instructions = {
            "high": "Proporciona una respuesta detallada y completa, basándote en la información disponible.",
            "medium": "Ofrece una respuesta equilibrada y matizada.",
            "low": "Responde de manera conversacional. Si hay incertidumbre, menciónalo naturalmente."
        }
        
        # Construir prompt estilo instruct
        prompt_lines = [
            "Eres regerIA, un asistente especializado en historia hispanoamericana.",
            "Tu objetivo es explicar procesos históricos con precisión y claridad.",
            "No hagas resúmenes extensos ni justifiques posturas políticas.",
            "Utiliza el contexto proporcionado solo si es directamente relevante para responder.",
            "Si el contexto no aplica, responde con tu conocimiento general.",
            f"{instructions[confidence]}",
            "Tu respuesta debe ser en español, con un estilo claro y accesible.",
            ""
        ]
        
        # Añadir contexto si existe
        if context and confidence != "low":
            prompt_lines.append("Información contextual para considerar:")
            prompt_lines.append(context)
            prompt_lines.append("")
        
        # Pregunta y formato claro
        prompt_lines.append(f"Pregunta del usuario: {question}")
        prompt_lines.append("")
        prompt_lines.append("Por favor, desarrolla una respuesta adecuada:")
        
        return "\n".join(prompt_lines)

    def _extract_gguf_response(self, response, original_prompt: str, original_question: str) -> str:
        """Extrae y limpia la respuesta de GGUF"""
        
        # 1. Obtener texto crudo
        if isinstance(response, dict) and "choices" in response:
            raw_text = response["choices"][0]["text"].strip()
        else:
            raw_text = str(response).strip()
        
        print(f"🔍 GGUF raw response length: {len(raw_text)} chars")
        
        # 2. Eliminar el prompt si está incluido (GGUF a veces lo repite)
        if original_prompt in raw_text:
            cleaned = raw_text.replace(original_prompt, "").strip()
            print(f"   → Prompt eliminado, quedan {len(cleaned)} chars")
        else:
            cleaned = raw_text
        
        # 3. Eliminar repeticiones de la pregunta
        if original_question in cleaned:
            cleaned = cleaned.replace(original_question, "").strip()
        
        # 4. Eliminar frases iniciales comunes
        start_phrases = [
            "La respuesta es:",
            "Respuesta:",
            "Basándome en la información:",
            "Según el contexto:",
            "Puedo responder que",
            "En primer lugar,",
            "Como regerIA,"
        ]
        
        for phrase in start_phrases:
            if cleaned.startswith(phrase):
                cleaned = cleaned[len(phrase):].strip()
                break
        
        # 5. Eliminar contenido duplicado o repetitivo
        import re
        # Eliminar repeticiones de "Eres regerIA"
        if "Eres regerIA" in cleaned:
            parts = cleaned.split("Eres regerIA")
            if len(parts) > 1:
                cleaned = parts[-1].strip()
        
        # 6. Limpiar espacios y saltos de línea excesivos
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
        cleaned = cleaned.strip()
        
        print(f"   → Respuesta final: {len(cleaned)} chars")
        
        # 7. Si es demasiado corta, intentar extraer más
        if len(cleaned) < 100 and len(raw_text) > 500:
            print("⚠️  Respuesta muy corta, intentando extracción alternativa...")
            # Buscar después de "respuesta" o "desarrolla"
            for keyword in ["desarrolla", "respuesta", "Por favor"]:
                if keyword in raw_text:
                    parts = raw_text.split(keyword)
                    if len(parts) > 1:
                        alternative = parts[-1].strip()
                        if len(alternative) > len(cleaned):
                            cleaned = alternative
                            print(f"   → Usando alternativa con {len(cleaned)} chars")
                            break
        
        return cleaned

    def generate_response(self, question: str, context_docs: List[Dict], max_chars: int = 2000) -> str:
        """Genera respuesta optimizada para GGUF y Transformers"""
        
        start_time = datetime.now()
        
        confidence = self.compute_confidence(context_docs)
        context = ""
        if confidence != "low":
            context = self.build_intelligent_context(question, context_docs)
        
        # ===== 1. CONSTRUIR PROMPT DIFERENTE SEGÚN TIPO =====
        if self.model_type == "gguf":
            prompt = self._build_gguf_prompt(question, context, confidence)
            print("🔧 Usando prompt optimizado para GGUF")
        else:
            prompt = self.build_prompt_with_confidence(question, context, confidence)
            print("🔧 Usando prompt estándar para Transformers")
        
        # DEBUG: Mostrar información del prompt
        print(f"🔍 Prompt length: {len(prompt)} chars")
        print(f"🔍 Model type: {self.model_type}")
        
        if confidence == "high":
            temperature = 0.6
        elif confidence == "medium":
            temperature = 0.7
        else:
            temperature = 0.85

        # Ajustar tokens según modelo
        model_name = self.model_info["name"]
        if "40b" in model_name.lower():
            max_new_tokens = 1200  # Aumentado para GGUF
            print(f"🔍 Max tokens configurados: {max_new_tokens} (esperados ~{max_new_tokens * 3} chars)")
        else:
            max_new_tokens = self.model_info["max_tokens"]

        # ===== 2. GENERACIÓN SEGÚN TIPO =====
        if self.model_type == "gguf":
            # GENERACIÓN GGUF OPTIMIZADA
            print(f"⚡ GGUF: generando hasta {max_new_tokens} tokens...")
            
            try:
                response = self.model(
                    prompt,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=TOP_P,
                    top_k=TOP_K,
                    repeat_penalty=1.1,  # 🔥 CLAVE para evitar repeticiones
                    stop=["</s>", "###", "\n\n", "Human:", "Usuario:", "Pregunta:", "Question:", "Instrucción:"],
                    echo=False,
                    stream=False
                )
                
            except Exception as e:
                print(f"❌ Error en generación GGUF: {e}")
                return "Error generando respuesta GGUF"
            
            # Extraer y limpiar respuesta GGUF
            response_text = self._extract_gguf_response(response, prompt, question)
            
        else:
            # GENERACIÓN TRANSFORMERS (tu código original)
            max_length = 3500 if "40b" in model_name.lower() else 2500
            
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            ).to(self.model.device)
            
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
            
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extraer solo la respuesta para Transformers
            if "RESPUESTA:" in response_text:
                response_text = response_text.split("RESPUESTA:")[-1].strip()
            elif "respuesta:" in response_text:
                response_text = response_text.split("respuesta:")[-1].strip()
        
        # ===== 3. POST-PROCESAMIENTO =====
        # Limitar longitud
        if len(response_text) > max_chars:
            if "." in response_text[max_chars-200:max_chars]:
                last_period = response_text[:max_chars].rfind(".")
                response_text = response_text[:last_period+1]
        
        # Estadísticas
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"✅ Respuesta en {elapsed:.1f}s, {len(response_text)} caracteres")
        
        # Limpiar memoria
        self.cleanup_memory()
        
        return response_text.strip()
    
    def cleanup_memory(self):
        """Limpia memoria GPU"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_model_info(self):
        """Obtiene información del modelo actual"""
        return self.model_info

    def unload_model(self):
        """Libera COMPLETAMENTE el modelo y VRAM (especialmente para GGUF)"""
        try:
            print("🧹 INICIANDO LIBERACIÓN COMPLETA DE VRAM...")
            
            # 1. Liberar modelo llama-cpp ESPECÍFICAMENTE
            if hasattr(self, 'model') and self.model is not None:
                if self.model_type == "gguf":
                    print("🔧 Liberando modelo GGUF (llama-cpp)...")
                    try:
                        # Método específico para llama-cpp
                        if hasattr(self.model, '_ctx'):
                            del self.model._ctx
                        if hasattr(self.model, 'ctx'):
                            del self.model.ctx
                        # Forzar eliminación
                        self.model.__del__()
                    except:
                        pass
                
                # Eliminar referencia
                del self.model
                self.model = None
            
            # 2. Liberar tokenizer y pipeline
            if hasattr(self, "tokenizer"):
                del self.tokenizer
                self.tokenizer = None
            
            if hasattr(self, "pipeline"):
                del self.pipeline
                self.pipeline = None
            
            # 3. LIMPIEZA AGGRESIVA DE VRAM CUDA
            if torch.cuda.is_available():
                print("🔥 Limpieza agresiva CUDA...")
                
                # Vaciar cachés
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Forzar recolección CUDA
                torch.cuda.ipc_collect()
                
                # Forzar recolección Python
                import gc
                gc.collect()
                
                # Esperar y limpiar de nuevo
                import time
                time.sleep(1)  # Dar tiempo
                torch.cuda.empty_cache()
                
                # Verificar
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                print(f"✅ VRAM liberada: {allocated:.1f}GB / {reserved:.1f}GB")
            
            self.model_loaded = False
            print("✅ Modelo completamente descargado")
            
        except Exception as e:
            print(f"⚠️ Error en liberación: {e}")
            # Intentar limpiar aunque falle
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
            n_ctx=2048,
            n_gpu_layers=-1,  # Todas las capas en GPU
            n_batch=512,
            n_threads=4,  # Hilos para procesamiento auxiliar
            verbose=True
        )
        
        self.model_type = "gguf"  # <-- IMPORTANTE: Establecer el tipo
        self.model_loaded = True
        print("✅ Modelo GGUF cargado")