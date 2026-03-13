# -*- coding: utf-8 -*-
"""Almacén vectorial con metadatos enriquecidos"""

import chromadb
import re
import unicodedata
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import json


def _normalize(text: str) -> str:
    """Elimina tildes y pasa a minúsculas para comparaciones."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


def _extract_keyword_terms(query: str) -> List[str]:
    """
    Extrae términos literales que deben aparecer en el chunk.
    Detecta referencias del tipo: 'artículo 11', 'capítulo IV', 'art. 3', etc.
    Devuelve variantes con/sin tilde para mayor cobertura.
    """
    terms = []
    q_norm = _normalize(query)

    # Artículos: "artículo 11", "articulo 11", "art. 11", "art 11"
    for m in re.finditer(r'\bart[i\.]?[c\.]?[u\.]?[l\.]?[o\.]?\s*\.?\s*(\d+)', q_norm):
        num = m.group(1)
        terms.append(f"Artículo {num}.")
        terms.append(f"Articulo {num}.")  # sin tilde por si acaso

    # Capítulos romanos o numéricos: "capítulo IV", "capitulo 3"
    for m in re.finditer(r'\bcap[i\.]?[t\.]?[u\.]?[l\.]?[o\.]?\s*\.?\s*([IVXivx\d]+)', q_norm):
        ref = m.group(1).upper()
        terms.append(f"CAPÍTULO {ref}")
        terms.append(f"Capítulo {ref}")

    return terms

class PersistentVectorStore:
    def __init__(self, persist_path: str):
        self.persist_path = persist_path
        
        self.client = chromadb.PersistentClient(
            path=persist_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        try:
            self.collection = self.client.get_collection(name="hispanidad_docs")
            print(f"📚 Colección cargada: {self.collection.name} ({self.collection.count()} chunks)")
        except:
            self.collection = self.client.create_collection(
                name="hispanidad_docs",
                metadata={"description": "Documentos históricos hispánicos con metadatos enriquecidos"}
            )
            print("🆕 Nueva colección creada")
    
    def _split_text_into_chunks(self, text: str, chunk_size: int = 1400, overlap: int = 200) -> List[str]:
        """
        Divide texto en chunks con solapamiento.
        Estrategia:
          1. Intenta separar por párrafos dobles (\n\n).
          2. Si quedan bloques > chunk_size, los subdivide por línea simple (\n).
          3. Aplica ventana deslizante con overlap para no perder contexto entre chunks.
        """
        # Paso 1: separar por \n\n
        raw_blocks = [b.strip() for b in text.split('\n\n') if b.strip()]

        # Paso 2: subdividir bloques demasiado grandes por \n simple
        units: List[str] = []
        for block in raw_blocks:
            if len(block) <= chunk_size:
                if len(block) >= 40:          # ignorar líneas muy cortas/vacías
                    units.append(block)
            else:
                lines = [l.strip() for l in block.split('\n') if l.strip() and len(l.strip()) >= 20]
                units.extend(lines)

        if not units:
            # Fallback: ventana deslizante pura sobre caracteres
            return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap) if text[i:i + chunk_size].strip()]

        # Paso 3: acumular con ventana deslizante
        chunks: List[str] = []
        current = ""
        for unit in units:
            if len(current) + len(unit) + 2 <= chunk_size:
                current = (current + "\n\n" + unit).strip() if current else unit
            else:
                if current:
                    chunks.append(current)
                # Solapamiento: llevar el final del chunk anterior al nuevo
                overlap_text = current[-overlap:] if len(current) > overlap else current
                current = (overlap_text + "\n\n" + unit).strip() if overlap_text else unit

        if current.strip():
            chunks.append(current.strip())

        return chunks

    def add_pdf_chunks(self, pdf_id: str, text: str, pdf_metadata: Dict, analysis: Dict) -> int:
        """
        Añade chunks CON metadatos enriquecidos del análisis.
        Usa chunking robusto con ventana deslizante para manejar
        documentos con cualquier formato de saltos de línea.
        """
        chunks_text = self._split_text_into_chunks(text, chunk_size=1400, overlap=200)

        if not chunks_text:
            print("   ⚠️  No se generaron chunks (texto vacío o demasiado corto)")
            return 0

        chunks = []
        metadatas = []
        ids = []

        for chunk_num, chunk_content in enumerate(chunks_text):
            chunk_metadata = {
                'pdf_id': pdf_id,
                'pdf_title': pdf_metadata.get('title', pdf_metadata.get('filename', '')),
                'pdf_author': pdf_metadata.get('author', ''),
                'pdf_pages': pdf_metadata.get('pages', 0),
                'chunk_num': chunk_num,
                'total_chunks': len(chunks_text),
                'type': 'historia_hispanica',
                'source': 'PDF',
                'quality': pdf_metadata.get('quality', 'media'),
                'document_themes': json.dumps(analysis.get('themes', [])),
                'document_summary': analysis.get('summary', '')[:200],
                'document_entities': json.dumps(analysis.get('entities', {})),
                'analysis_version': analysis.get('analysis_version', '1.0'),
                'has_full_analysis': True
            }
            chunks.append(chunk_content)
            metadatas.append(chunk_metadata)
            ids.append(f"{pdf_id}_chunk_{chunk_num}")

        if chunks:
            self.collection.add(
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
            print(f"   📝 Añadidos {len(chunks)} chunks con metadatos enriquecidos")

        return len(chunks)
    
    def _format_result(self, doc: str, metadata: Dict, distance: float) -> Dict:
        """Convierte un resultado de ChromaDB al formato interno."""
        try:
            themes = json.loads(metadata.get('document_themes', '[]'))
            summary = metadata.get('document_summary', '')
            has_analysis = metadata.get('has_full_analysis', False)
        except Exception:
            themes, summary, has_analysis = [], '', False

        l2_dist = distance
        base_score = max(0.0, 1.0 - (l2_dist ** 2) / 2.0)
        if has_analysis:
            base_score = min(1.0, base_score + 0.05)

        return {
            'text': doc,
            'metadata': metadata,
            'enriched_metadata': {
                'themes': themes,
                'summary': summary,
                'has_full_analysis': has_analysis
            },
            'score': base_score,
            'pdf_title': metadata.get('pdf_title', 'Sin título')
        }

    def search_with_analysis(self, query: str, n_results: int = 4, use_themes: bool = True) -> List[Dict]:
        """
        Búsqueda híbrida: vectorial + búsqueda literal de texto.
        Si la consulta menciona artículos o capítulos concretos, busca
        primero los chunks que los contienen literalmente y los antepone
        a los resultados vectoriales.
        """
        seen_texts: set = set()
        merged: List[Dict] = []

        def _add(item: Dict, score_override: Optional[float] = None) -> None:
            key = item['text'][:120]
            if key not in seen_texts:
                seen_texts.add(key)
                if score_override is not None:
                    item = {**item, 'score': score_override}
                merged.append(item)

        # ── 1. Búsqueda literal para referencias exactas ───────────────────
        keyword_terms = _extract_keyword_terms(query)
        for term in keyword_terms:
            try:
                kw_results = self.collection.get(
                    where_document={"$contains": term},
                    include=["documents", "metadatas"]
                )
                if kw_results['documents']:
                    for doc, meta in zip(kw_results['documents'], kw_results['metadatas']):
                        item = self._format_result(doc, meta, distance=0.0)
                        item['score'] = 1.0  # máxima prioridad: coincidencia exacta
                        _add(item)
                    print(f"   🔎 Búsqueda literal '{term}': {len(kw_results['documents'])} chunks")
            except Exception as e:
                print(f"   ⚠️  Búsqueda literal '{term}' fallida: {e}")

        # ── 2. Búsqueda vectorial (semántica) ──────────────────────────────
        try:
            vec_n = max(n_results * 2, 12)
            results = self.collection.query(
                query_texts=[query],
                n_results=vec_n,
                include=["documents", "metadatas", "distances"]
            )

            if results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    meta = results['metadatas'][0][i]
                    dist = results['distances'][0][i] if results['distances'] else 0
                    item = self._format_result(doc, meta, dist)
                    _add(item)

        except Exception as e:
            print(f"❌ Error en búsqueda vectorial: {e}")

        # ── 3. Ordenar y devolver ──────────────────────────────────────────
        merged.sort(key=lambda x: x['score'], reverse=True)
        return merged[:n_results]

    def get_stats(self) -> Dict:
        """Obtiene estadísticas del almacén vectorial"""
        try:
            count = self.collection.count()

            all_metas = self.collection.get(include=["metadatas"])
            pdf_ids = set()
            if all_metas['metadatas']:
                for meta in all_metas['metadatas']:
                    if meta and 'pdf_id' in meta:
                        pdf_ids.add(meta['pdf_id'])

            return {
                'total_chunks': count,
                'unique_pdfs': len(pdf_ids),
                'path': self.persist_path
            }
        except:
            return {'total_chunks': 0, 'unique_pdfs': 0}