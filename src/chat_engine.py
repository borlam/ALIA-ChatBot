# -*- coding: utf-8 -*-
"""Motor optimizado para Salamandra-7B"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from .config import *

class ChatEngine:
    def __init__(self):
        print("🧠 Cargando modelo salamandra-7b (optimizado para memoria)...")
        
        # CONFIGURACIÓN DE CUANTIZACIÓN 4-BIT (ESENCIAL)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,  # ← CRÍTICO: Reduce memoria 4x
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,  # ← Aplicar cuantización
            device_map="auto",  # Distribución automática
            torch_dtype=torch.float16,
            trust_remote_code=False
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("🔤 Cargando embeddings...")
        # Cargar embeddings en CPU para ahorrar VRAM
        self.embedder = SentenceTransformer(EMBEDDING_MODEL, device='cpu')
        
        # Mover a GPU solo cuando sea necesario
        self.embedder_on_gpu = False
        
        print(f"✅ Modelo cargado en 4-bit")
        print(f"📊 Memoria GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    def move_embedder_to_gpu(self):
        """Mueve el embedder a GPU solo cuando sea necesario"""
        if not self.embedder_on_gpu and torch.cuda.is_available():
            self.embedder = self.embedder.to(torch.device("cuda"))
            self.embedder_on_gpu = True

    def generate_response(self, question: str, context_docs: List[Dict], max_chars: int = 1500) -> str:
        """Genera respuesta optimizada para memoria"""
        
        # 1. PROMPT MÁS INTELIGENTE Y CORTO
        context_text = ""
        pdf_sources = {}
        
        # Contexto muy limitado para 7B
        for i, doc in enumerate(context_docs[:1]):  # SOLO 1 documento para 7B
            source = doc.get('pdf_title', 'Documento')
            if source not in pdf_sources:
                pdf_sources[source] = []
            pdf_sources[source].append(i + 1)
            
            # Solo 200 caracteres máximo
            context_text += f"\n[Fuente: {source}]: {doc['text'][:200]}..."
        
        # Prompt adaptativo según longitud de pregunta
        question_words = len(question.split())
        
        if question_words < 5:
            # Preguntas cortas
            prompt = f"""Responde brevemente a esta pregunta sobre historia hispánica:

Pregunta: {question}

Contexto: {context_text if context_text else "No hay contexto específico."}

Respuesta concisa:"""
            max_gen_tokens = 200
        else:
            # Preguntas complejas
            prompt = f"""### Instrucción:
Como historiador experto en cultura hispánica, responde de forma clara y precisa.

### Contexto disponible:
{context_text if context_text else "Consulta general sobre historia hispánica."}

### Pregunta:
{question}

### Respuesta fundamentada:"""
            max_gen_tokens = 400
        
        # 2. TOKENIZACIÓN OPTIMIZADA
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=768  # Reducido de 1024
        ).to(self.model.device)
        
        # 3. GENERACIÓN CON LÍMITES ESTRICTOS
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_gen_tokens,  # Usar límite adaptativo
                min_new_tokens=50,
                temperature=0.7,  # Más bajo para menos alucinaciones
                do_sample=True,
                top_p=0.9,
                top_k=30,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                length_penalty=1.0,  # Neutral
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
        
        # 4. DECODIFICACIÓN Y LIMPIEZA
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extraer solo la respuesta
        if "Respuesta:" in response:
            response = response.split("Respuesta:")[-1].strip()
        elif "respuesta:" in response:
            response = response.split("respuesta:")[-1].strip()
        
        # Limpiar encabezados del prompt
        for marker in ["### Instrucción:", "### Contexto:", "### Pregunta:", "Instrucción:", "Contexto:"]:
            if marker in response:
                response = response.split(marker)[0].strip()
        
        # 5. POST-PROCESAMIENTO INTELIGENTE
        import re
        
        # Eliminar URLs largas (ahorran espacio)
        response = re.sub(r'https?://\S+', '[referencia]', response)
        
        # Limitar longitud
        if len(response) > max_chars:
            # Buscar punto de corte natural
            if "." in response[max_chars-100:max_chars]:
                last_period = response[:max_chars].rfind(".")
                response = response[:last_period+1]
            else:
                response = response[:max_chars] + "..."
        
        # 6. AÑADIR FUENTES (opcional)
        if pdf_sources and context_docs:
            sources = list(pdf_sources.keys())
            if sources:
                response += f"\n\n📄 Fuente: {sources[0]}"
        
        print(f"📝 Respuesta: {len(response)} caracteres")
        
        # Limpiar caché de GPU después de cada generación
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return response.strip()
    
    def cleanup(self):
        """Limpia memoria GPU"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 Memoria GPU limpiada")
