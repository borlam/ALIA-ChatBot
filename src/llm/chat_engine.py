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
    
    def build_intelligent_context(self, question: str, documents: List[Dict]) -> str:
        """
        Construye contexto INTELIGENTE usando análisis pre-existente
        """
        if not documents:
            return "No hay documentos relevantes para esta pregunta."
        
        context_parts = []
        
        # 1. DOCUMENTOS MÁS RELEVANTES (ya vienen ordenados por score)
        context_parts.append("### 📚 INFORMACIÓN DOCUMENTAL RELEVANTE:")
        
        for i, doc in enumerate(documents[:3]):  # Máximo 3 documentos
            source = doc.get('pdf_title', f'Documento {i+1}')
            text_preview = doc['text'][:300] + "..." if len(doc['text']) > 300 else doc['text']
            
            # Usar metadatos enriquecidos si están disponibles
            enriched = doc.get('enriched_metadata', {})
            
            context_parts.append(f"\n**{source}**")
            context_parts.append(f"*Contenido relevante:* {text_preview}")
            
            if enriched.get('has_full_analysis'):
                themes = enriched.get('themes', [])
                if themes:
                    context_parts.append(f"*Temas del documento:* {', '.join(themes[:3])}")
            
            # Separador
            if i < len(documents[:3]) - 1:
                context_parts.append("---")
        
        return '\n'.join(context_parts)
    
    def build_optimized_prompt(self, question: str, context: str) -> str:
        """Prompt optimizado y eficiente"""
        
        return f"""Eres regerIA, especialista en historia hispanoamericana.

CONTEXTO DOCUMENTAL DISPONIBLE:
{context}

INSTRUCCIONES:
1. Responde de forma clara y precisa a la pregunta
2. Usa información específica de los documentos cuando sea relevante
3. Complementa con tu conocimiento general cuando sea útil
4. Si un documento menciona algo específico, cítalo brevemente
5. Sé conciso pero completo

PREGUNTA: {question}

RESPUESTA:"""
    
    def generate_response(self, question: str, context_docs: List[Dict], max_chars: int = 2000) -> str:
        """Genera respuesta RÁPIDA usando análisis pre-existente"""
        
        start_time = datetime.now()
        
        # 1. Construir contexto INTELIGENTE (RÁPIDO)
        context = self.build_intelligent_context(question, context_docs)
        
        # 2. Prompt optimizado
        prompt = self.build_optimized_prompt(question, context)
        
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
                temperature=0.7,
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