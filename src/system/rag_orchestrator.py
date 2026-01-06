# -*- coding: utf-8 -*-
"""Orquestador principal del sistema RAG"""

from typing import Dict, List, Any
import torch
from datetime import datetime
from ..processing.pdf_manager import PDFManager
from ..vector.vector_store import PersistentVectorStore
from ..llm.chat_engine import ChatEngine
from ..core.document_analyzer import DocumentAnalyzer
from .config import *
from datetime import datetime

class RAGOrchestrator:
    """Coordina todos los componentes del sistema"""
    
    def __init__(self, initial_model_key: str = None):
        print("\n" + "="*60)
        print("🏛️  INICIALIZANDO SISTEMA RAG (ARQUITECTURA OPTIMIZADA)")
        print("="*60)
        
        # Establecer modelo inicial si se especifica
        if initial_model_key:
            set_active_model(initial_model_key)
        
        # Componentes
        self.pdf_manager = PDFManager(PDF_STORAGE_PATH)
        self.vector_store = PersistentVectorStore(VECTOR_DB_PATH)
        self.document_analyzer = DocumentAnalyzer()
        self.chat_engine = ChatEngine(ACTIVE_MODEL_KEY)  # Pasar modelo activo
        
        # Estado del sistema
        self.system_stats = self._update_stats()
        
        print("\n✅ SISTEMA OPTIMIZADO LISTO")
        print(f"🤖 Modelo activo: {get_active_model_info()['display_name']}")
        print(f"📊 Documentos: {self.system_stats.get('total_pdfs', 0)}")
        print(f"📊 Chunks: {self.system_stats.get('total_chunks', 0):,}")
        print(f"⚡ Arquitectura: Análisis en indexación")
        print(f"💾 GPU: {'✅ ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else '❌ Solo CPU'}")
    
    def _update_stats(self) -> Dict:
        """Actualiza estadísticas del sistema"""
        pdf_stats = self.pdf_manager.get_stats()
        vector_stats = self.vector_store.get_stats()
        
        return {
            **pdf_stats,
            **vector_stats,
            'gpu_available': torch.cuda.is_available(),
            'model': get_active_model_info(),
            'architecture': 'optimized_v2',
            'last_update': datetime.now().isoformat()
        }
    
    def change_model(self, model_key: str) -> Dict[str, Any]:
        """Cambia el modelo de lenguaje activo"""
        print(f"\n🔄 SOLICITUD DE CAMBIO DE MODELO: {model_key}")
        
        try:
            # Verificar si el modelo es válido
            if model_key not in get_available_models_list():
                return {
                    'success': False,
                    'error': f"Modelo '{model_key}' no disponible. Opciones: {list(get_available_models_list().keys())}"
                }
            
            # Verificar GPU para modelos grandes
            if "40b" in model_key.lower() and not is_gpu_sufficient_for_model(model_key):
                return {
                    'success': False,
                    'error': f"⚠️ ALIA-40B requiere ~20GB de GPU. Tu GPU tiene {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB"
                }
            
            # Cambiar configuración
            if not set_active_model(model_key):
                return {
                    'success': False,
                    'error': "Error al cambiar configuración del modelo"
                }
            
            # Crear nuevo ChatEngine
            print("🔧 Recargando motor de chat...")
            self.chat_engine = ChatEngine(model_key)
            
            # Actualizar estadísticas
            self.system_stats = self._update_stats()
            
            return {
                'success': True,
                'model': get_active_model_info(),
                'message': f"✅ Modelo cambiado a {get_active_model_info()['display_name']}"
            }
            
        except Exception as e:
            print(f"❌ Error cambiando modelo: {e}")
            return {
                'success': False,
                'error': f"Error al cambiar modelo: {str(e)}"
            }
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Obtiene lista de modelos disponibles"""
        models = get_available_models_list()
        
        # Añadir información de compatibilidad con GPU
        enhanced_models = {}
        for key, info in models.items():
            enhanced_models[key] = {
                **info,
                'gpu_sufficient': is_gpu_sufficient_for_model(key),
                'is_current': key == ACTIVE_MODEL_KEY
            }
        
        return enhanced_models
    
    def get_current_model_info(self) -> Dict[str, Any]:
        """Obtiene información del modelo actual"""
        info = get_active_model_info()
        info['gpu_sufficient'] = is_gpu_sufficient_for_model()
        return info
    
    def process_document(self, pdf_file, filename: str = None) -> Dict:
        """
        Procesa documento COMPLETO con análisis incluido
        """
        # 1. Extraer y analizar (PDFManager ya lo hace)
        result = self.pdf_manager.process_pdf(pdf_file, filename)
        
        if not result['success']:
            return result
        
        # 2. Añadir a vector store CON análisis
        pdf_id = result['pdf_id']
        text = result['text']
        metadata = result['metadata']
        analysis = result.get('analysis', {})
        
        print(f"📝 Indexando documento con metadatos enriquecidos...")
        chunks_added = self.vector_store.add_pdf_chunks(
            pdf_id, text, metadata, analysis
        )
        
        # 3. Actualizar registro
        if chunks_added > 0:
            self.pdf_manager.processed_log[pdf_id]['chunks_generated'] = chunks_added
            self.pdf_manager.processed_log[pdf_id]['status'] = 'indexed'
            self.pdf_manager.save_processed_log()
        
        # 4. Actualizar estadísticas
        self.system_stats = self._update_stats()
        
        return {
            'success': True,
            'pdf_id': pdf_id,
            'filename': filename,
            'chunks_added': chunks_added,
            'analysis_done': True,
            'document_themes': analysis.get('themes', [])[:3],
            'total_pdfs': self.system_stats['total_pdfs']
        }
    
    def query(self, question: str, max_docs: int = 3) -> Dict:
        """
        Consulta inteligente con análisis pre-existente
        """
        print(f"\n🔍 CONSULTA: '{question[:80]}...'")
        print(f"🤖 Modelo: {get_active_model_info()['display_name']}")
        
        # 1. Búsqueda MEJORADA con metadatos enriquecidos
        search_results = self.vector_store.search_with_analysis(
            query=question,
            n_results=max_docs * 2,  # Traer más para filtrar
            use_themes=True
        )
        
        # 2. Filtrar y ordenar
        relevant_docs = search_results[:max_docs]
        
        print(f"   📚 Documentos relevantes encontrados: {len(relevant_docs)}")
        
        # 3. Generar respuesta (RÁPIDO)
        response = self.chat_engine.generate_response(
            question=question,
            context_docs=relevant_docs,
            max_chars=1800
        )
        
        # 4. Preparar respuesta estructurada
        sources = []
        for doc in relevant_docs:
            source_info = {
                'title': doc.get('pdf_title', 'Documento'),
                'score': f"{doc.get('score', 0):.2f}",
                'has_analysis': doc.get('enriched_metadata', {}).get('has_full_analysis', False)
            }
            sources.append(source_info)
        
        return {
            'question': question,
            'answer': response,
            'sources': sources,
            'docs_used': len(relevant_docs),
            'response_length': len(response),
            'model': get_active_model_info()['display_name']
        }
    
    def get_system_info(self) -> Dict:
        """Obtiene información completa del sistema"""
        return self.system_stats
    
    def search_by_theme(self, theme: str) -> List[Dict]:
        """Busca documentos por tema (usando análisis previo)"""
        return self.pdf_manager.search_by_theme(theme)

    def reload_llm(self, model_key: str):
        """Recarga completamente el LLM"""
        if hasattr(self, "chat_engine") and self.chat_engine:
            self.chat_engine.unload_model()
            self.chat_engine = None

        set_active_model(model_key)
        self.chat_engine = ChatEngine(model_key)
