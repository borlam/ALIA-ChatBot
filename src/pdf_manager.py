# -*- coding: utf-8 -*-
"""Gestión de PDFs y almacenamiento persistente"""

import os
import json
import hashlib
import time
from datetime import datetime
from typing import Dict, List, Optional
from .pdf_extractor import SmartPDFExtractor

class PDFManager:
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.extractor = SmartPDFExtractor()
        self.processed_log = {}
        self.load_processed_log()

    def load_processed_log(self) -> None:
        """Carga el registro de PDFs procesados"""
        log_path = os.path.join(self.storage_path, "processed_log.json")
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    self.processed_log = json.load(f)
                print(f"📋 Log cargado: {len(self.processed_log)} PDFs")
            except:
                self.processed_log = {}

    def save_processed_log(self) -> None:
        """Guarda el registro de PDFs procesados"""
        log_path = os.path.join(self.storage_path, "processed_log.json")
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.processed_log, f, indent=2, ensure_ascii=False)

    def save_pdf_file(self, pdf_bytes: bytes, original_filename: str) -> str:
        """Guarda un PDF en el almacenamiento"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = ''.join(c for c in original_filename if c.isalnum() or c in ' ._-')
        unique_name = f"{timestamp}_{safe_name}"
        save_path = os.path.join(self.storage_path, unique_name)

        with open(save_path, 'wb') as f:
            f.write(pdf_bytes)

        return save_path

    def process_pdf(self, pdf_bytes: bytes, original_filename: str) -> Dict:
        """Procesa un PDF y extrae su contenido"""
        print(f"\n{'='*50}")
        print(f"📤 PROCESANDO: {original_filename}")
        print(f"{'='*50}")

        temp_path = os.path.join(self.storage_path, f"temp_{int(time.time())}.pdf")
        with open(temp_path, 'wb') as f:
            f.write(pdf_bytes)

        try:
            text, metadata, quality = self.extractor.extract_text(temp_path)

            if not text or len(text) < 100:
                os.remove(temp_path)
                return {
                    'success': False,
                    'error': 'PDF sin texto extraíble',
                    'filename': original_filename
                }

            content_hash = hashlib.md5(text.encode()).hexdigest()

            for pdf_info in self.processed_log.values():
                if pdf_info.get('content_hash') == content_hash:
                    print(f"⏭️  PDF ya procesado")
                    os.remove(temp_path)
                    return {
                        'success': False,
                        'error': 'PDF duplicado',
                        'filename': original_filename
                    }

            final_path = self.save_pdf_file(pdf_bytes, original_filename)

            pdf_info = {
                'filename': original_filename,
                'stored_name': os.path.basename(final_path),
                'path': final_path,
                'content_hash': content_hash,
                'text_length': len(text),
                'pages': metadata.get('pages', '?'),
                'quality': quality,
                'author': metadata.get('author', ''),
                'title': metadata.get('title', ''),
                'processed_date': datetime.now().isoformat(),
                'chunks_generated': 0,
                'status': 'processed'
            }

            pdf_id = f"pdf_{len(self.processed_log) + 1:04d}"
            self.processed_log[pdf_id] = pdf_info
            self.save_processed_log()

            print(f"✅ PDF procesado exitosamente")
            print(f"   📄 Páginas: {pdf_info['pages']}")
            print(f"   📊 Caracteres: {pdf_info['text_length']:,}")

            return {
                'success': True,
                'pdf_id': pdf_id,
                'text': text,
                'metadata': pdf_info,
                'original_filename': original_filename
            }

        except Exception as e:
            print(f"❌ Error: {e}")
            return {
                'success': False,
                'error': str(e),
                'filename': original_filename
            }
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def get_pdf_stats(self) -> Dict:
        """Obtiene estadísticas de los PDFs procesados"""
        total_pdfs = len(self.processed_log)
        total_pages = sum(info.get('pages', 0) for info in self.processed_log.values() 
                         if isinstance(info.get('pages'), int))
        total_chars = sum(info.get('text_length', 0) for info in self.processed_log.values())

        quality_counts = {}
        for info in self.processed_log.values():
            quality = info.get('quality', 'desconocida')
            quality_counts[quality] = quality_counts.get(quality, 0) + 1

        return {
            'total_pdfs': total_pdfs,
            'total_pages': total_pages,
            'total_chars': f"{total_chars:,}",
            'quality_distribution': quality_counts,
            'last_update': datetime.now().strftime("%Y-%m-%d %H:%M")
        }
