# -*- coding: utf-8 -*-
"""Almacén vectorial con metadatos enriquecidos"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
import json

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
    
    def search_with_analysis(self, query: str, n_results: int = 4, use_themes: bool = True) -> List[Dict]:
        """
        Búsqueda mejorada que usa metadatos de análisis
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results * 2,  # Traer más para filtrar
                include=["documents", "metadatas", "distances"]
            )
            
            # Procesar resultados CON análisis
            formatted = []
            if results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i]
                    
                    # Extraer metadatos enriquecidos
                    try:
                        themes = json.loads(metadata.get('document_themes', '[]'))
                        summary = metadata.get('document_summary', '')
                        has_analysis = metadata.get('has_full_analysis', False)
                    except:
                        themes = []
                        summary = ''
                        has_analysis = False
                    
                    # Calcular score mejorado
                    base_score = 1 - (results['distances'][0][i] if results['distances'] else 0)
                    
                    # Bonus por documentos con análisis completo
                    if has_analysis:
                        base_score += 0.1
                    
                    formatted.append({
                        'text': doc[:600] + "..." if len(doc) > 600 else doc,
                        'metadata': metadata,
                        'enriched_metadata': {
                            'themes': themes,
                            'summary': summary,
                            'has_full_analysis': has_analysis
                        },
                        'score': base_score,
                        'pdf_title': metadata.get('pdf_title', 'Sin título')
                    })
            
            # Ordenar por score y limitar
            formatted.sort(key=lambda x: x['score'], reverse=True)
            
            return formatted[:n_results]
            
        except Exception as e:
            print(f"❌ Error en búsqueda mejorada: {e}")
            return []

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