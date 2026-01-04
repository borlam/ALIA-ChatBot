# -*- coding: utf-8 -*-
"""Orquestador principal del sistema RAG"""

from typing import Dict, List, Any
from ..processing.pdf_manager import PDFManager
from ..vector.vector_store import PersistentVectorStore
from ..llm.chat_engine import ChatEngine
from ..core.document_analyzer import DocumentAnalyzer
from .config import *

class RAGOrchestrator:
    """Coordina todos los componentes del sistema"""
    
    def __init__(self):
        print("\n" + "="*60)
        print("🏛️  INICIALIZANDO SISTEMA RAG (ARQUITECTURA OPTIMIZADA)")
        print("="*60)
        
        # Componentes
        self.pdf_manager = PDFManager(PDF_STORAGE_PATH)
        self.vector_store = PersistentVectorStore(VECTOR_DB_PATH)
        self.document_analyzer = DocumentAnalyzer()
        self.chat_engine = ChatEngine()
        
        # Estado del sistema
        self.system_stats = self._update_stats()
        
        print("\n✅ SISTEMA OPTIMIZADO LISTO")
        print(f"📊 Documentos: {self.system_stats.get('total_pdfs', 0)}")
        print(f"📊 Chunks: {self.system_stats.get('total_chunks', 0):,}")
        print(f"⚡ Arquitectura: Análisis en indexación")
    
    def _update_stats(self) -> Dict:
        """Actualiza estadísticas del sistema"""
        pdf_stats = self.pdf_manager.get_pdf_stats()
        vector_stats = self.vector_store.get_stats()
        
        return {
            **pdf_stats,
            **vector_stats,
            'gpu_available': torch.cuda.is_available(),
            'model': MODEL_NAME,
            'architecture': 'optimized_v2',
            'last_update': datetime.now().isoformat()
        }
    
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
            'model': MODEL_NAME
        }
    
    def get_system_info(self) -> Dict:
        """Obtiene información completa del sistema"""
        return self.system_stats
    
    def search_by_theme(self, theme: str) -> List[Dict]:
        """Busca documentos por tema (usando análisis previo)"""
        return self.pdf_manager.search_by_theme(theme)