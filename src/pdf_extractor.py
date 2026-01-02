def extract_with_pymupdf(self, pdf_path: str) -> Tuple[str, Dict]:
    text = ""
    metadata = {}

    try:
        doc = fitz.open(pdf_path)
        metadata = {
            'pages': len(doc),
            'author': doc.metadata.get('author', ''),
            'title': doc.metadata.get('title', ''),
            'subject': doc.metadata.get('subject', ''),
            'method': 'PyMuPDF'
        }

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()
            if page_text.strip():
                text += f"\n\n--- Página {page_num + 1} ---\n{page_text}"

        doc.close()
        return text.strip(), metadata

    except Exception as e:
        print(f"⚠️ PyMuPDF falló: {e}")
        return "", {}

def extract_with_pdfplumber(self, pdf_path: str) -> Tuple[str, Dict]:
    text = ""

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n\n--- Página {page_num + 1} ---\n{page_text}"

        return text.strip(), {'pages': len(pdf.pages), 'method': 'pdfplumber'}

    except Exception as e:
        print(f"⚠️ pdfplumber falló: {e}")
        return "", {}

def extract_with_ocr(self, pdf_path: str) -> Tuple[str, Dict]:
    """Extrae texto usando OCR - para PDFs escaneados/imagen"""
    if not OCR_AVAILABLE:
        print("⚠️ OCR no disponible. Instala: pip install pytesseract pillow")
        return "", {}
    
    text = ""
    
    try:
        # Abrir con PyMuPDF para obtener imágenes
        doc = fitz.open(pdf_path)
        metadata = {
            'pages': len(doc),
            'author': doc.metadata.get('author', ''),
            'title': doc.metadata.get('title', ''),
            'subject': doc.metadata.get('subject', ''),
            'method': 'OCR (PDF escaneado)'
        }
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Convertir página a imagen de alta resolución
            pix = page.get_pixmap(dpi=300)
            img_data = pix.tobytes("png")
            
            # Usar PIL para abrir la imagen
            img = Image.open(io.BytesIO(img_data))
            
            # Aplicar OCR
            page_text = pytesseract.image_to_string(img, lang='spa')  # Español
            
            if page_text.strip():
                text += f"\n\n--- Página {page_num + 1} ---\n{page_text}"
        
        doc.close()
        return text.strip(), metadata
        
    except Exception as e:
        print(f"⚠️ OCR falló: {e}")
        return "", {}

def extract_text(self, pdf_path: str) -> Tuple[str, Dict, str]:
    """Extrae texto de un PDF usando múltiples métodos"""
    print(f"📖 Extrayendo: {os.path.basename(pdf_path)}")

    # Intentar PyMuPDF primero
    text1, meta1 = self.extract_with_pymupdf(pdf_path)
    if text1 and len(text1) > 100:
        text_len = len(text1)
        print(f"✅ PyMuPDF: {text_len:,} caracteres, {meta1.get('pages', '?')} páginas")

        avg_words = len(text1.split()) / max(1, meta1.get('pages', 1))
        quality = "baja" if avg_words < 50 else "media" if avg_words < 200 else "alta"

        return text1, meta1, quality

    # Fallback a pdfplumber
    text2, meta2 = self.extract_with_pdfplumber(pdf_path)
    if text2 and len(text2) > 100:
        text_len = len(text2)
        print(f"✅ pdfplumber: {text_len:,} caracteres, {meta2.get('pages', '?')} páginas")

        avg_words = len(text2.split()) / max(1, meta2.get('pages', 1))
        quality = "baja" if avg_words < 50 else "media" if avg_words < 200 else "alta"

        return text2, meta2, quality

    # Fallback a OCR (si está disponible)
    if OCR_AVAILABLE:
        print("🔄 Intentando OCR...")
        text3, meta3 = self.extract_with_ocr(pdf_path)
        if text3 and len(text3) > 50:
            text_len = len(text3)
            print(f"✅ OCR exitoso: {text_len:,} caracteres")
            quality = "ocr"  # Calidad especial para OCR
            return text3, meta3, quality
    else:
        print("ℹ️ OCR no disponible para intentar extracción de PDFs escaneados")

    print(f"❌ Todos los métodos fallaron")
    return "", {}, "error"

def _calculate_quality(self, text: str, metadata: Dict) -> str:
    """Calcula la calidad de la extracción"""
    if not metadata.get('pages'):
        return "desconocida"
        
    avg_words = len(text.split()) / max(1, metadata['pages'])
    if avg_words < 50:
        return "baja"
    elif avg_words < 200:
        return "media"
    else:
        return "alta"
