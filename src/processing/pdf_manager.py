# -*- coding: utf-8 -*-
"""Gestor de PDFs CON análisis integrado y OCR mejorado"""

import os
import json
import hashlib
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

# Importar el nuevo analizador y extractor mejorado
from ..core.document_analyzer import DocumentAnalyzer
from .pdf_extractor import SmartPDFExtractor

class PDFManager:
    def __init__(self, storage_path: str, use_ocr: bool = True, ocr_lang: str = 'spa'):
        """
        Args:
            storage_path: Ruta para almacenar PDFs
            use_ocr: Habilitar OCR para PDFs escaneados
            ocr_lang: Idioma para OCR
        """
        self.storage_path = storage_path
        # Usar el extractor mejorado con OCR
        self.extractor = SmartPDFExtractor(use_ocr=use_ocr, ocr_lang=ocr_lang)
        self.analyzer = DocumentAnalyzer()
        self.processed_log = {}
        
        # Crear directorio si no existe
        os.makedirs(storage_path, exist_ok=True)
        
        self.load_processed_log()
        print(f"📁 PDF Manager inicializado en: {storage_path}")
        print(f"   OCR: {'Habilitado' if use_ocr else 'Deshabilitado'}")
        print(f"   Idioma OCR: {ocr_lang}")
    
    def load_processed_log(self):
        """Carga el registro con análisis incluidos"""
        log_path = os.path.join(self.storage_path, "processed_log.json")
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    self.processed_log = json.load(f)
                print(f"📋 Log cargado: {len(self.processed_log)} PDFs procesados")
            except Exception as e:
                print(f"⚠️ Error cargando log: {e}")
                self.processed_log = {}
        else:
            print("📋 No se encontró log previo, creando nuevo")
            self.processed_log = {}
    
    def save_processed_log(self):
        """Guarda el registro con análisis"""
        log_path = os.path.join(self.storage_path, "processed_log.json")
        try:
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(self.processed_log, f, indent=2, ensure_ascii=False, default=str)
            print(f"💾 Log guardado: {len(self.processed_log)} PDFs")
        except Exception as e:
            print(f"❌ Error guardando log: {e}")
    
    def save_pdf_file(self, pdf_bytes: bytes, original_filename: str) -> str:
        """Guarda el archivo PDF en disco con nombre único"""
        # Crear nombre de archivo seguro
        safe_name = self._create_safe_filename(original_filename)
        
        # Si ya existe, añadir timestamp
        counter = 1
        base_name, ext = os.path.splitext(safe_name)
        
        while True:
            if counter == 1:
                filename = safe_name
            else:
                filename = f"{base_name}_{counter}{ext}"
            
            filepath = os.path.join(self.storage_path, filename)
            if not os.path.exists(filepath):
                break
            counter += 1
        
        # Guardar archivo
        with open(filepath, 'wb') as f:
            f.write(pdf_bytes)
        
        return filepath
    
    def _create_safe_filename(self, filename: str) -> str:
        """Crea un nombre de archivo seguro"""
        # Mantener la extensión
        name, ext = os.path.splitext(filename)
        
        # Reemplazar caracteres no seguros
        safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in name)
        
        # Limitar longitud
        if len(safe_name) > 100:
            safe_name = safe_name[:100]
        
        return safe_name.strip() + ext.lower()
    
    def process_pdf(self, pdf_bytes: bytes, original_filename: str) -> Dict[str, Any]:
        """Procesa PDF CON análisis completo y OCR mejorado"""
        print(f"\n{'='*60}")
        print(f"📤 PROCESANDO Y ANALIZANDO: {original_filename}")
        print(f"{'='*60}")
        
        # Crear archivo temporal
        temp_path = os.path.join(self.storage_path, f"temp_{int(time.time())}_{hashlib.md5(pdf_bytes).hexdigest()[:8]}.pdf")
        
        try:
            # Guardar bytes en archivo temporal
            with open(temp_path, 'wb') as f:
                f.write(pdf_bytes)
            
            # 1. ANÁLISIS PREVIO DEL PDF
            print("   🔍 Analizando tipo de PDF...")
            pdf_analysis = self.extractor.analyze_pdf_type(temp_path)
            
            print(f"   📊 Tipo: {'📄 Texto' if pdf_analysis['has_text_layer'] else '🖼️ Escaneado'}")
            print(f"   📄 Páginas estimadas: {pdf_analysis['estimated_pages']}")
            print(f"   💾 Tamaño: {pdf_analysis['file_size_kb']:.1f} KB")
            print(f"   🎯 Método sugerido: {pdf_analysis['suggested_method']}")
            
            # 2. EXTRACCIÓN DE TEXTO
            print("   📖 Extrayendo texto...")
            text, metadata, quality = self.extractor.extract_text(temp_path)
            
            if not text or len(text.strip()) < 50:
                error_msg = "PDF sin texto extraíble"
                if pdf_analysis['is_scanned'] and not self.extractor.use_ocr:
                    error_msg += " (PDF escaneado, active OCR)"
                elif pdf_analysis['is_scanned']:
                    error_msg += " (OCR falló)"
                
                print(f"❌ {error_msg}")
                return {
                    'success': False,
                    'error': error_msg,
                    'filename': original_filename,
                    'analysis': pdf_analysis
                }
            
            # 3. ANÁLISIS SEMÁNTICO DEL CONTENIDO
            print("   🔬 Analizando contenido semántico...")
            content_analysis = self.analyzer.analyze_complete_document(text, original_filename)
            
            # 4. VERIFICAR DUPLICADOS (por hash de análisis y contenido)
            analysis_hash = hashlib.md5(
                f"{content_analysis.get('themes_hash', '')}:{content_analysis.get('summary_hash', '')}"
                .encode()
            ).hexdigest()
            
            duplicate_info = self._check_duplicate(analysis_hash, original_filename)
            if duplicate_info['is_duplicate']:
                print(f"⏭️  PDF duplicado detectado: {duplicate_info['reason']}")
                return {
                    'success': False,
                    'error': f'PDF duplicado: {duplicate_info["reason"]}',
                    'filename': original_filename,
                    'duplicate_of': duplicate_info['pdf_id']
                }
            
            # 5. GUARDAR ARCHIVO PERMANENTE
            print("   💾 Guardando archivo...")
            final_path = self.save_pdf_file(pdf_bytes, original_filename)
            
            # 6. CREAR REGISTRO COMPLETO
            pdf_info = {
                'filename': original_filename,
                'stored_name': os.path.basename(final_path),
                'path': final_path,
                'text': text,
                'analysis': content_analysis,
                'pdf_analysis': pdf_analysis,  # Análisis del tipo de PDF
                'extraction_metadata': metadata,
                'analysis_hash': analysis_hash,
                'text_length': len(text),
                'pages': metadata.get('pages', pdf_analysis.get('estimated_pages', 0)),
                'quality': quality,
                'extraction_method': metadata.get('method', 'unknown'),
                'author': metadata.get('author', ''),
                'title': metadata.get('title', original_filename),
                'processed_date': datetime.now().isoformat(),
                'chunks_generated': 0,
                'status': 'analyzed'
            }
            
            pdf_id = f"pdf_{len(self.processed_log) + 1:04d}"
            self.processed_log[pdf_id] = pdf_info
            self.save_processed_log()
            
            # 7. MOSTRAR RESUMEN
            print(f"\n✅ PDF PROCESADO EXITOSAMENTE")
            print(f"   📋 ID: {pdf_id}")
            print(f"   📄 Método: {metadata.get('method', 'unknown')}")
            print(f"   📊 Calidad: {quality}")
            print(f"   🔤 Caracteres: {len(text):,}")
            print(f"   🎯 Temas principales: {', '.join(content_analysis.get('themes', [])[:3])}")
            
            if 'summary' in content_analysis:
                summary = content_analysis['summary']
                if len(summary) > 150:
                    summary = summary[:150] + "..."
                print(f"   📝 Resumen: {summary}")
            
            return {
                'success': True,
                'pdf_id': pdf_id,
                'text': text,
                'analysis': content_analysis,
                'pdf_analysis': pdf_analysis,
                'metadata': pdf_info,
                'original_filename': original_filename
            }
            
        except Exception as e:
            print(f"❌ Error en procesamiento: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'success': False,
                'error': str(e),
                'filename': original_filename
            }
            
        finally:
            # Limpiar archivo temporal
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
    
    def _check_duplicate(self, analysis_hash: str, filename: str) -> Dict[str, Any]:
        """Verifica si el PDF es un duplicado"""
        for pdf_id, pdf_info in self.processed_log.items():
            # 1. Verificar por hash de análisis
            if pdf_info.get('analysis_hash') == analysis_hash:
                return {
                    'is_duplicate': True,
                    'reason': 'contenido idéntico',
                    'pdf_id': pdf_id
                }
            
            # 2. Verificar por nombre de archivo (ignorando mayúsculas)
            if pdf_info['filename'].lower() == filename.lower():
                return {
                    'is_duplicate': True,
                    'reason': 'mismo nombre de archivo',
                    'pdf_id': pdf_id
                }
        
        return {'is_duplicate': False}
    
    def get_document_analysis(self, pdf_id: str) -> Optional[Dict]:
        """Obtiene el análisis completo de un documento"""
        pdf_info = self.processed_log.get(pdf_id)
        if pdf_info:
            return {
                'content_analysis': pdf_info.get('analysis', {}),
                'pdf_analysis': pdf_info.get('pdf_analysis', {}),
                'metadata': {k: v for k, v in pdf_info.items() 
                           if k not in ['text', 'analysis', 'pdf_analysis']}
            }
        return None
    
    def search_by_theme(self, theme: str, min_relevance: float = 0.3) -> List[Dict]:
        """Busca documentos por tema usando análisis previo"""
        results = []
        theme_lower = theme.lower()
        
        for pdf_id, pdf_info in self.processed_log.items():
            analysis = pdf_info.get('analysis', {})
            themes = analysis.get('themes', [])
            theme_weights = analysis.get('theme_weights', {})
            
            relevance = 0
            found_in = []
            
            # Buscar en temas principales
            for doc_theme in themes:
                doc_theme_lower = doc_theme.lower()
                if theme_lower == doc_theme_lower:
                    relevance = max(relevance, 1.0)  # Coincidencia exacta
                    found_in.append(f"tema: {doc_theme}")
                elif theme_lower in doc_theme_lower:
                    relevance = max(relevance, 0.7)  # Contenido en tema
                    found_in.append(f"tema: {doc_theme}")
                elif any(word in doc_theme_lower for word in theme_lower.split()):
                    relevance = max(relevance, 0.4)  # Palabra en tema
                    found_in.append(f"tema: {doc_theme}")
            
            # Buscar en palabras clave
            keywords = analysis.get('keywords', [])
            for keyword in keywords:
                keyword_lower = keyword.lower()
                if theme_lower == keyword_lower:
                    relevance = max(relevance, 0.6)
                    found_in.append(f"keyword: {keyword}")
                elif theme_lower in keyword_lower:
                    relevance = max(relevance, 0.3)
                    found_in.append(f"keyword: {keyword}")
            
            # Buscar en resumen
            summary = analysis.get('summary', '').lower()
            if theme_lower in summary:
                words_in_summary = sum(1 for word in theme_lower.split() if word in summary)
                relevance = max(relevance, words_in_summary / max(len(theme_lower.split()), 1) * 0.5)
                found_in.append("resumen")
            
            if relevance >= min_relevance:
                results.append({
                    'pdf_id': pdf_id,
                    'title': pdf_info.get('title', pdf_info['filename']),
                    'filename': pdf_info['filename'],
                    'themes': themes[:5],
                    'summary': analysis.get('summary', '')[:200] + ('...' if len(analysis.get('summary', '')) > 200 else ''),
                    'relevance': relevance,
                    'found_in': found_in[:3],
                    'pages': pdf_info.get('pages', 0),
                    'quality': pdf_info.get('quality', 'unknown')
                })
        
        # Ordenar por relevancia
        results.sort(key=lambda x: x['relevance'], reverse=True)
        return results
    
    def get_all_documents(self) -> List[Dict]:
        """Obtiene lista de todos los documentos procesados"""
        documents = []
        for pdf_id, pdf_info in self.processed_log.items():
            documents.append({
                'pdf_id': pdf_id,
                'filename': pdf_info['filename'],
                'title': pdf_info.get('title', pdf_info['filename']),
                'pages': pdf_info.get('pages', 0),
                'text_length': pdf_info.get('text_length', 0),
                'quality': pdf_info.get('quality', 'unknown'),
                'processed_date': pdf_info.get('processed_date', ''),
                'themes': pdf_info.get('analysis', {}).get('themes', [])[:3]
            })
        
        # Ordenar por fecha más reciente primero
        documents.sort(key=lambda x: x.get('processed_date', ''), reverse=True)
        return documents
    
    def delete_document(self, pdf_id: str) -> bool:
        """Elimina un documento del sistema"""
        if pdf_id not in self.processed_log:
            return False
        
        try:
            pdf_info = self.processed_log[pdf_id]
            filepath = pdf_info.get('path')
            
            # Eliminar archivo físico
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
            
            # Eliminar del registro
            del self.processed_log[pdf_id]
            self.save_processed_log()
            
            print(f"🗑️ Documento {pdf_id} eliminado: {pdf_info['filename']}")
            return True
            
        except Exception as e:
            print(f"❌ Error eliminando documento {pdf_id}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del gestor de PDFs"""
        total_docs = len(self.processed_log)
        total_text = 0
        total_pages = 0
        quality_counts = {}
        extraction_methods = {}
        
        for pdf_info in self.processed_log.values():
            total_text += pdf_info.get('text_length', 0)
            total_pages += pdf_info.get('pages', 0)
            
            quality = pdf_info.get('quality', 'unknown')
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
            
            method = pdf_info.get('extraction_method', 'unknown')
            extraction_methods[method] = extraction_methods.get(method, 0) + 1
        
        return {
            'total_documents': total_docs,
            'total_text_characters': total_text,
            'total_pages': total_pages,
            'avg_pages_per_doc': total_pages / max(total_docs, 1),
            'quality_distribution': quality_counts,
            'extraction_methods': extraction_methods
        }