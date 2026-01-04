# -*- coding: utf-8 -*-
"""Extracción inteligente de texto de PDFs con OCR mejorado"""

import fitz  # PyMuPDF
import pdfplumber
import os
import logging
from typing import Tuple, Dict, Optional

# Configurar logging
logger = logging.getLogger(__name__)

class SmartPDFExtractor:
    def __init__(self, use_ocr: bool = True, ocr_lang: str = 'spa'):
        """
        Args:
            use_ocr: Si es True, usa OCR cuando no hay texto extraíble
            ocr_lang: Idioma para OCR ('spa' para español, 'eng' para inglés)
        """
        self.use_ocr = use_ocr
        self.ocr_lang = ocr_lang
        self.ocr_available = self._check_ocr_availability()
        
        print(f"📄 Extractor de PDFs inicializado (OCR: {'✅' if self.ocr_available and use_ocr else '❌'})")
        if self.ocr_available and use_ocr:
            print(f"   Idioma OCR: {ocr_lang}")
    
    def _check_ocr_availability(self) -> bool:
        """Verifica si las dependencias de OCR están disponibles"""
        try:
            import pytesseract
            from PIL import Image
            import io
            return True
        except ImportError:
            return False
    
    def extract_with_pymupdf(self, pdf_path: str) -> Tuple[str, Dict]:
        """Extrae texto usando PyMuPDF"""
        text = ""
        metadata = {}
        
        try:
            doc = fitz.open(pdf_path)
            metadata = {
                'pages': len(doc),
                'author': doc.metadata.get('author', ''),
                'title': doc.metadata.get('title', ''),
                'subject': doc.metadata.get('subject', ''),
                'method': 'PyMuPDF',
                'has_text_layer': True
            }
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                if page_text.strip():
                    text += f"\n\n--- Página {page_num + 1} ---\n{page_text}"
            
            doc.close()
            
            # Verificar si realmente tiene texto
            if text.strip():
                return text.strip(), metadata
            else:
                metadata['has_text_layer'] = False
                return "", metadata
                
        except Exception as e:
            logger.warning(f"PyMuPDF falló: {e}")
            return "", {}
    
    def extract_with_pdfplumber(self, pdf_path: str) -> Tuple[str, Dict]:
        """Extrae texto usando pdfplumber"""
        text = ""
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n\n--- Página {page_num + 1} ---\n{page_text}"
                
                metadata = {
                    'pages': pages,
                    'method': 'pdfplumber',
                    'has_text_layer': bool(text.strip())
                }
                
                return text.strip(), metadata
                
        except Exception as e:
            logger.warning(f"pdfplumber falló: {e}")
            return "", {}
    
    def extract_with_ocr(self, pdf_path: str) -> Tuple[str, Dict]:
        """Extrae texto usando OCR - versión mejorada"""
        if not self.ocr_available or not self.use_ocr:
            logger.warning("OCR no disponible o desactivado")
            return "", {}
        
        try:
            import pytesseract
            from PIL import Image
            import io
            
            # Abrir con PyMuPDF para obtener imágenes
            doc = fitz.open(pdf_path)
            metadata = {
                'pages': len(doc),
                'author': doc.metadata.get('author', ''),
                'title': doc.metadata.get('title', ''),
                'subject': doc.metadata.get('subject', ''),
                'method': 'OCR',
                'has_text_layer': False,
                'ocr_language': self.ocr_lang
            }
            
            text = ""
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Convertir página a imagen de alta resolución
                zoom = 2  # 200% de zoom para mejor calidad OCR
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img_data = pix.tobytes("png")
                
                # Usar PIL para abrir la imagen
                img = Image.open(io.BytesIO(img_data))
                
                # Configuración mejorada para OCR
                custom_config = f'--oem 3 --psm 6 -l {self.ocr_lang}'
                
                # Aplicar OCR
                page_text = pytesseract.image_to_string(img, config=custom_config)
                
                if page_text.strip():
                    text += f"\n\n--- Página {page_num + 1} ---\n{page_text}"
                
                # Limpiar memoria
                del pix, img
            
            doc.close()
            
            if text.strip():
                return text.strip(), metadata
            else:
                return "", metadata
                
        except Exception as e:
            logger.error(f"OCR falló: {e}")
            return "", {}
    
    def analyze_pdf_type(self, pdf_path: str) -> Dict:
        """Analiza el tipo de PDF para determinar el mejor método"""
        analysis = {
            'has_text_layer': False,
            'is_scanned': True,
            'suggested_method': 'unknown',
            'file_size_kb': 0,
            'estimated_pages': 0
        }
        
        try:
            # Obtener tamaño del archivo
            analysis['file_size_kb'] = os.path.getsize(pdf_path) / 1024
            
            # Intentar extraer con PyMuPDF para análisis
            doc = fitz.open(pdf_path)
            analysis['estimated_pages'] = len(doc)
            
            # Verificar si tiene texto
            has_text = False
            sample_text = ""
            
            # Verificar primeras 3 páginas
            check_pages = min(3, len(doc))
            for i in range(check_pages):
                page_text = doc[i].get_text()
                if page_text and len(page_text.strip()) > 10:
                    has_text = True
                    sample_text = page_text
                    break
            
            doc.close()
            
            analysis['has_text_layer'] = has_text
            
            # Determinar método sugerido
            if has_text:
                analysis['is_scanned'] = False
                analysis['suggested_method'] = 'direct'
                
                # Si el texto es muy corto, podría necesitar OCR
                if len(sample_text.strip()) < 50:
                    analysis['suggested_method'] = 'hybrid'
            else:
                analysis['is_scanned'] = True
                if self.ocr_available and self.use_ocr:
                    analysis['suggested_method'] = 'ocr'
                else:
                    analysis['suggested_method'] = 'failed'
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analizando PDF: {e}")
            return analysis
    
    def extract_text(self, pdf_path: str) -> Tuple[str, Dict, str]:
        """Extrae texto de un PDF usando estrategia inteligente"""
        filename = os.path.basename(pdf_path)
        print(f"\n📖 Procesando: {filename}")
        
        # 1. Analizar tipo de PDF
        analysis = self.analyze_pdf_type(pdf_path)
        print(f"   📊 Análisis: {'Texto' if analysis['has_text_layer'] else 'Escaneado'}, "
              f"{analysis['estimated_pages']} páginas, "
              f"{analysis['file_size_kb']:.1f} KB")
        
        # 2. Estrategia basada en análisis
        if analysis['suggested_method'] == 'direct':
            # PDF con texto, usar extracción directa
            text, metadata = self.extract_with_pymupdf(pdf_path)
            if not text:
                text, metadata = self.extract_with_pdfplumber(pdf_path)
            
            quality = self._calculate_quality(text, metadata)
            method_used = metadata.get('method', 'unknown')
            
        elif analysis['suggested_method'] == 'hybrid':
            # PDF con poco texto, intentar extracción directa primero
            text, metadata = self.extract_with_pymupdf(pdf_path)
            if not text or len(text.strip()) < 100:
                # Si la extracción directa falla o es pobre, usar OCR
                print("   🔄 Poco texto encontrado, intentando OCR...")
                ocr_text, ocr_meta = self.extract_with_ocr(pdf_path)
                if ocr_text and len(ocr_text) > len(text):
                    text = ocr_text
                    metadata = ocr_meta
                    method_used = 'hybrid (OCR)'
                else:
                    method_used = 'hybrid (direct)'
            else:
                method_used = 'hybrid (direct)'
            
            quality = self._calculate_quality(text, metadata)
            
        elif analysis['suggested_method'] == 'ocr':
            # PDF escaneado, usar OCR directamente
            text, metadata = self.extract_with_ocr(pdf_path)
            if text:
                method_used = 'ocr'
                quality = 'ocr'
            else:
                text = ""
                metadata = {}
                method_used = 'failed'
                quality = 'error'
                
        else:  # 'failed'
            text = ""
            metadata = {}
            method_used = 'failed'
            quality = 'error'
        
        # 3. Resultados
        if text and len(text.strip()) > 50:
            text_len = len(text)
            page_count = metadata.get('pages', analysis['estimated_pages'])
            print(f"✅ {method_used.upper()}: {text_len:,} caracteres, {page_count} páginas")
            print(f"   Calidad: {quality}")
            
            # Actualizar metadata con método usado
            if metadata:
                metadata['extraction_method'] = method_used
                metadata['quality'] = quality
            
            return text, metadata, quality
        else:
            print(f"❌ No se pudo extraer texto del PDF")
            return "", {}, "error"
    
    def _calculate_quality(self, text: str, metadata: Dict) -> str:
        """Calcula la calidad de la extracción"""
        if not text:
            return "error"
        
        pages = metadata.get('pages', 1)
        if pages == 0:
            pages = 1
        
        avg_words = len(text.split()) / pages
        
        if avg_words < 30:
            return "muy baja"
        elif avg_words < 100:
            return "baja"
        elif avg_words < 250:
            return "media"
        elif avg_words < 500:
            return "alta"
        else:
            return "excelente"
    
    def get_extraction_report(self, pdf_path: str) -> Dict:
        """Genera un reporte detallado de la extracción"""
        report = {}
        
        # Análisis del PDF
        report['analysis'] = self.analyze_pdf_type(pdf_path)
        
        # Extraer texto
        text, metadata, quality = self.extract_text(pdf_path)
        
        report['extraction'] = {
            'text_length': len(text),
            'quality': quality,
            'method': metadata.get('extraction_method', 'unknown'),
            'metadata': metadata
        }
        
        # Estadísticas del texto
        if text:
            words = text.split()
            report['text_stats'] = {
                'total_words': len(words),
                'total_characters': len(text),
                'unique_words': len(set(words)),
                'avg_words_per_page': len(words) / max(1, metadata.get('pages', 1))
            }
        
        return report