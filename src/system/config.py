# -*- coding: utf-8 -*-
"""Configuración centralizada del sistema"""

import os

# Rutas
DRIVE_PATH = "/content/drive/MyDrive/RAG_Hispanidad"
VECTOR_DB_PATH = f"{DRIVE_PATH}/vector_db"
PDF_STORAGE_PATH = f"{DRIVE_PATH}/pdf_storage"

# Modelos
#MODEL_NAME = "BSC-LT/salamandra-2b-instruct"
MODEL_NAME = "BSC-LT/salamandra-7b-instruct"
# Para usar 7B: os.environ["MODEL_SIZE"] = "BSC-LT/salamandra-7b-instruct"
# EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Configuración de chunks
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 100
MAX_CHUNKS_PER_PDF = 1000

# Configuración de generación
MAX_TOKENS = 600
MIN_TOKENS = 80
TEMPERATURE = 0.5  # ¡Reducido para precisión!
TOP_P = 0.9
TOP_K = 40

# Interfaz
DEFAULT_RESPONSE_LENGTH = 2000
MAX_RESPONSE_LENGTH = 5000

# Crear directorios si no existen
def setup_directories():
    """Crea la estructura de directorios necesaria"""
    directories = [DRIVE_PATH, VECTOR_DB_PATH, PDF_STORAGE_PATH]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Directorio creado/verificado: {directory}")
