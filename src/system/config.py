# -*- coding: utf-8 -*-
"""Configuración centralizada del sistema"""

import os
from typing import Dict, Any

# Rutas
DRIVE_PATH = "/content/drive/MyDrive/RAG_Hispanidad"
VECTOR_DB_PATH = f"{DRIVE_PATH}/vector_db"
PDF_STORAGE_PATH = f"{DRIVE_PATH}/pdf_storage"

# ===== MODELOS DISPONIBLES =====
AVAILABLE_MODELS = {
    "salamandra2b": {
        "name": "BSC-LT/salamandra-2b-instruct",
        "display_name": "Salamandra 2B",
        "description": "Modelo ligero y rápido",
        "memory_required": "2-3 GB",
        "max_tokens": 512
    },
    "salamandra7b": {
        "name": "BSC-LT/salamandra-7b-instruct",
        "display_name": "Salamandra 7B", 
        "description": "Modelo equilibrado (por defecto)",
        "memory_required": "4-6 GB",
        "max_tokens": 600
    },
    "alia40b": {
        "name": "BSC-LT/ALIA-40b-instruct",
        "display_name": "ALIA 40B",
        "description": "Modelo avanzado multilingüe",
        "memory_required": "20-24 GB",
        "max_tokens": 800
    }
}

# Modelo activo
ACTIVE_MODEL_KEY = "alia40b"
MODEL_NAME = AVAILABLE_MODELS[ACTIVE_MODEL_KEY]["name"]

# Configuración de embeddings
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Configuración de chunks
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 100
MAX_CHUNKS_PER_PDF = 1000

# Configuración de generación (se ajustará según modelo)
MAX_TOKENS = 600
MIN_TOKENS = 80
TEMPERATURE = 0.5
TOP_P = 0.9
TOP_K = 40

# Interfaz
DEFAULT_RESPONSE_LENGTH = 2000
MAX_RESPONSE_LENGTH = 5000

# ===== FUNCIONES DE MODELOS =====

def set_active_model(model_key: str) -> bool:
    """Cambia el modelo activo del sistema"""
    global ACTIVE_MODEL_KEY, MODEL_NAME, MAX_TOKENS
    
    if model_key not in AVAILABLE_MODELS:
        print(f"❌ Modelo '{model_key}' no disponible. Modelos disponibles: {list(AVAILABLE_MODELS.keys())}")
        return False
    
    ACTIVE_MODEL_KEY = model_key
    MODEL_NAME = AVAILABLE_MODELS[model_key]["name"]
    MAX_TOKENS = AVAILABLE_MODELS[model_key]["max_tokens"]
    
    print(f"✅ Modelo cambiado a: {AVAILABLE_MODELS[model_key]['display_name']}")
    print(f"   📝 Descripción: {AVAILABLE_MODELS[model_key]['description']}")
    print(f"   💾 Memoria requerida: {AVAILABLE_MODELS[model_key]['memory_required']}")
    
    return True

def get_active_model_info() -> Dict[str, Any]:
    """Obtiene información del modelo activo"""
    model_info = AVAILABLE_MODELS[ACTIVE_MODEL_KEY].copy()
    model_info["key"] = ACTIVE_MODEL_KEY
    return model_info

def get_available_models_list() -> Dict[str, Dict[str, str]]:
    """Lista todos los modelos disponibles"""
    return AVAILABLE_MODELS

def is_gpu_sufficient_for_model(model_key: str = None) -> bool:
    """Verifica si la GPU tiene suficiente memoria para el modelo"""
    import torch
    
    if not torch.cuda.is_available():
        return False
    
    model_key = model_key or ACTIVE_MODEL_KEY
    memory_required = AVAILABLE_MODELS[model_key]["memory_required"]
    
    # Extraer número de GB requeridos
    import re
    match = re.search(r'(\d+)-(\d+)', memory_required)
    if match:
        min_gb = int(match.group(1))
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        # Dejar 2GB para el sistema
        available_gb = gpu_memory_gb - 2
        
        return available_gb >= min_gb
    
    return True  # Si no podemos determinar, asumimos que sí

# ===== FUNCIONES EXISTENTES =====

def setup_directories():
    """Crea la estructura de directorios necesaria"""
    directories = [DRIVE_PATH, VECTOR_DB_PATH, PDF_STORAGE_PATH]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Directorio creado/verificado: {directory}")