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
    
    def add_pdf_chunks(self, pdf_id: str, text: str, pdf_metadata: Dict, analysis: Dict) -> int:
        """
        Añade chunks CON metadatos enriquecidos del análisis
        
        Args:
            analysis: Análisis completo del documento desde DocumentAnalyzer
        """
        # 1. Dividir texto en chunks (lógica existente)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 100]
        
        chunks = []
        metadatas = []
        ids = []
        
        current_chunk = ""
        chunk_num = 0
        
        for para in paragraphs:
            if len(current_chunk) + len(para) < 1500:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            else:
                if current_chunk:
                    # 2. METADATOS ENRIQUECIDOS con análisis
                    chunk_metadata = {
                        'pdf_id': pdf_id,
                        'pdf_title': pdf_metadata.get('title', pdf_metadata.get('filename', '')),
                        'pdf_author': pdf_metadata.get('author', ''),
                        'pdf_pages': pdf_metadata.get('pages', 0),
                        'chunk_num': chunk_num,
                        'total_chunks': 0,
                        'type': 'historia_hispanica',
                        'source': 'PDF',
                        'quality': pdf_metadata.get('quality', 'media'),
                        
                        # METADATOS ENRIQUECIDOS DEL ANÁLISIS
                        'document_themes': json.dumps(analysis.get('themes', [])),
                        'document_summary': analysis.get('summary', '')[:200],
                        'document_entities': json.dumps(analysis.get('entities', {})),
                        'analysis_version': analysis.get('analysis_version', '1.0'),
                        'has_full_analysis': True
                    }
                    
                    chunks.append(current_chunk)
                    metadatas.append(chunk_metadata)
                    ids.append(f"{pdf_id}_chunk_{chunk_num}")
                    
                    chunk_num += 1
                    current_chunk = para
        
        # ... (resto de la lógica de chunks) ...
        
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