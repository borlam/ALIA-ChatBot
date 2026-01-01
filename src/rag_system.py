# -*- coding: utf-8 -*-
"""Sistema RAG principal que coordina todos los componentes"""

from datetime import datetime
from typing import List, Dict
from .pdf_manager import PDFManager
from .vector_store import PersistentVectorStore
from .chat_engine import ChatEngine
from .config import *

class PDFRAGSystem:
    def __init__(self):
        print("\n" + "="*60)
        print("🏛️  INICIALIZANDO SISTEMA RAG CON PDFs")
        print("="*60)

        self.pdf_manager = PDFManager(PDF_STORAGE_PATH)
        self.vector_store = PersistentVectorStore(VECTOR_DB_PATH)
        self.chat_engine = ChatEngine()

        self.update_stats()

        print("\n✅ SISTEMA LISTO")
        print(f"📚 PDFs: {self.pdf_stats['total_pdfs']}")
        print(f"📝 Chunks: {self.vector_stats['total_chunks']}")

    def update_stats(self) -> None:
        """Actualiza las estadísticas del sistema"""
        self.pdf_stats = self.pdf_manager.get_pdf_stats()
        self.vector_stats = self.vector_store.get_stats()
        self.system_stats = {
            **self.pdf_stats,
            **self.vector_stats,
            'gpu': torch.cuda.is_available(),
            'model': MODEL_NAME,
            'last_update': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def upload_and_process_pdf(self, pdf_file, filename: str = None) -> Dict:
        """Sube y procesa un PDF"""
        if pdf_file is None:
            return {'success': False, 'error': 'No se seleccionó archivo'}

        try:
            if hasattr(pdf_file, 'read'):
                pdf_bytes = pdf_file.read()
                actual_filename = filename or getattr(pdf_file, 'name', 'documento.pdf')
            else:
                with open(pdf_file, 'rb') as f:
                    pdf_bytes = f.read()
                actual_filename = filename or os.path.basename(pdf_file)

            print(f"📤 Subiendo: {actual_filename}")

            result = self.pdf_manager.process_pdf(pdf_bytes, actual_filename)

            if not result['success']:
                return result

            pdf_id = result['pdf_id']
            text = result['text']
            metadata = result['metadata']

            print(f"📝 Generando chunks...")
            chunks_added = self.vector_store.add_pdf_chunks(pdf_id, text, metadata)

            if chunks_added > 0:
                self.pdf_manager.processed_log[pdf_id]['chunks_generated'] = chunks_added
                self.pdf_manager.processed_log[pdf_id]['status'] = 'indexed'
                self.pdf_manager.save_processed_log()

            self.update_stats()

            return {
                'success': True,
                'pdf_id': pdf_id,
                'filename': actual_filename,
                'chunks_added': chunks_added,
                'pages': metadata.get('pages', 0),
                'text_length': metadata.get('text_length', 0),
                'quality': metadata.get('quality', 'media'),
                'total_pdfs': self.system_stats['total_pdfs'],
                'total_chunks': self.system_stats['total_chunks']
            }

        except Exception as e:
            print(f"❌ Error: {e}")
            return {'success': False, 'error': str(e)}

    def search_documents(self, query: str, n_results: int = 4) -> List[Dict]:
        """Busca documentos relevantes"""
        return self.vector_store.search(query, n_results)

    def generate_response(self, question: str, context_docs: List[Dict], max_chars: int = 2500) -> str:
        """Genera respuesta usando el chat engine"""
        return self.chat_engine.generate_response(question, context_docs, max_chars)

    def get_system_info(self) -> Dict:
        """Obtiene información del sistema"""
        return self.system_stats
