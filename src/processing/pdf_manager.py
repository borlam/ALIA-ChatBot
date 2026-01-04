# -*- coding: utf-8 -*-
"""Gestor de PDFs CON análisis integrado"""

import os
import json
import hashlib
import time
from datetime import datetime
from typing import Dict, List, Optional

# Importar el nuevo analizador
from ..core.document_analyzer import DocumentAnalyzer
from .pdf_extractor import SmartPDFExtractor

class PDFManager:
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.extractor = SmartPDFExtractor()
        self.analyzer = DocumentAnalyzer()  # NUEVO: Analizador
        self.processed_log = {}
        self.load_processed_log()
    
    def load_processed_log(self):
        """Carga el registro con análisis incluidos"""
        log_path = os.path.join(self.storage_path, "processed_log.json")
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    self.processed_log = json.load(f)
                print(f"📋 Log cargado: {len(self.processed_log)} PDFs")
            except:
                self.processed_log = {}
    
    def save_processed_log(self):
        """Guarda el registro con análisis"""
        log_path = os.path.join(self.storage_path, "processed_log.json")
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(self.processed_log, f, indent=2, ensure_ascii=False)
    
    def process_pdf(self, pdf_bytes: bytes, original_filename: str) -> Dict:
        """Procesa PDF CON análisis completo"""
        print(f"\n{'='*50}")
        print(f"📤 PROCESANDO Y ANALIZANDO: {original_filename}")
        print(f"{'='*50}")
        
        # ... (código existente para extraer texto) ...
        
        # 1. Extraer texto (código existente)
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
            
            # 2. ANÁLISIS COMPLETO DEL DOCUMENTO (UNA SOLA VEZ)
            print("   🔍 Analizando documento (esto se hace una sola vez)...")
            analysis = self.analyzer.analyze_complete_document(text, original_filename)
            
            # 3. Verificar duplicados por hash de análisis
            analysis_hash = hashlib.md5(str(analysis).encode()).hexdigest()
            
            for pdf_info in self.processed_log.values():
                if pdf_info.get('analysis_hash') == analysis_hash:
                    print(f"⏭️  PDF ya procesado (contenido similar)")
                    os.remove(temp_path)
                    return {
                        'success': False,
                        'error': 'PDF duplicado',
                        'filename': original_filename
                    }
            
            # 4. Guardar archivo
            final_path = self.save_pdf_file(pdf_bytes, original_filename)
            
            # 5. Crear registro COMPLETO con análisis
            pdf_info = {
                'filename': original_filename,
                'stored_name': os.path.basename(final_path),
                'path': final_path,
                'text': text,  # Guardar texto completo para chunks
                'analysis': analysis,  # NUEVO: Análisis completo guardado
                'analysis_hash': analysis_hash,
                'text_length': len(text),
                'pages': metadata.get('pages', '?'),
                'quality': quality,
                'author': metadata.get('author', ''),
                'title': metadata.get('title', original_filename),
                'processed_date': datetime.now().isoformat(),
                'chunks_generated': 0,
                'status': 'analyzed'  # Nuevo estado
            }
            
            pdf_id = f"pdf_{len(self.processed_log) + 1:04d}"
            self.processed_log[pdf_id] = pdf_info
            self.save_processed_log()
            
            print(f"✅ PDF procesado y analizado")
            print(f"   📊 Temas identificados: {', '.join(analysis['themes'][:3])}")
            print(f"   📄 Resumen: {analysis['summary'][:100]}...")
            
            return {
                'success': True,
                'pdf_id': pdf_id,
                'text': text,
                'analysis': analysis,  # Incluir análisis en el resultado
                'metadata': pdf_info,
                'original_filename': original_filename
            }
            
        except Exception as e:
            print(f"❌ Error en procesamiento: {e}")
            return {
                'success': False,
                'error': str(e),
                'filename': original_filename
            }
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def get_document_analysis(self, pdf_id: str) -> Optional[Dict]:
        """Obtiene el análisis completo de un documento"""
        pdf_info = self.processed_log.get(pdf_id)
        if pdf_info and 'analysis' in pdf_info:
            return pdf_info['analysis']
        return None
    
    def search_by_theme(self, theme: str) -> List[Dict]:
        """Busca documentos por tema (usando análisis previo)"""
        results = []
        theme_lower = theme.lower()
        
        for pdf_id, pdf_info in self.processed_log.items():
            analysis = pdf_info.get('analysis', {})
            themes = analysis.get('themes', [])
            
            # Buscar tema en temas del documento
            for doc_theme in themes:
                if theme_lower in doc_theme.lower():
                    results.append({
                        'pdf_id': pdf_id,
                        'title': pdf_info.get('title', ''),
                        'themes': themes,
                        'summary': analysis.get('summary', ''),
                        'relevance': 2  # Peso para coincidencia exacta
                    })
                    break
        
        return results