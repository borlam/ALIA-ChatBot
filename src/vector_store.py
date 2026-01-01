# -*- coding: utf-8 -*-
"""Almacén vectorial persistente con ChromaDB"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict

class PersistentVectorStore:
    def __init__(self, persist_path: str):
        self.persist_path = persist_path

        self.client = chromadb.PersistentClient(
            path=persist_path,
            settings=Settings(anonymized_telemetry=False)
        )

        try:
            self.collection = self.client.get_collection(name="hispanidad_docs")
            print(f"📚 Colección cargada: {self.collection.name} ({self.collection.count()} documentos)")
        except:
            self.collection = self.client.create_collection(
                name="hispanidad_docs",
                metadata={"description": "Documentos históricos hispánicos"}
            )
            print("🆕 Nueva colección creada")

    def add_pdf_chunks(self, pdf_id: str, text: str, pdf_metadata: Dict) -> int:
        """Divide y añade chunks de texto al almacén vectorial"""
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
                    chunk_metadata = {
                        'pdf_id': pdf_id,
                        'pdf_title': pdf_metadata.get('title', pdf_metadata.get('filename', '')),
                        'pdf_author': pdf_metadata.get('author', ''),
                        'pdf_pages': pdf_metadata.get('pages', 0),
                        'chunk_num': chunk_num,
                        'total_chunks': 0,
                        'type': 'historia_hispanica',
                        'source': 'PDF',
                        'quality': pdf_metadata.get('quality', 'media')
                    }

                    chunks.append(current_chunk)
                    metadatas.append(chunk_metadata)
                    ids.append(f"{pdf_id}_chunk_{chunk_num}")

                    chunk_num += 1
                    current_chunk = para

        if current_chunk and len(chunks) < 1000:
            chunk_metadata = {
                'pdf_id': pdf_id,
                'pdf_title': pdf_metadata.get('title', pdf_metadata.get('filename', '')),
                'pdf_author': pdf_metadata.get('author', ''),
                'pdf_pages': pdf_metadata.get('pages', 0),
                'chunk_num': chunk_num,
                'total_chunks': chunk_num + 1,
                'type': 'historia_hispanica',
                'source': 'PDF',
                'quality': pdf_metadata.get('quality', 'media')
            }

            chunks.append(current_chunk)
            metadatas.append(chunk_metadata)
            ids.append(f"{pdf_id}_chunk_{chunk_num}")
            chunk_num += 1

        for metadata in metadatas:
            metadata['total_chunks'] = chunk_num

        if chunks:
            self.collection.add(
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
            print(f"   📝 Añadidos {len(chunks)} chunks")

        return len(chunks)

    def search(self, query: str, n_results: int = 4) -> List[Dict]:
        """Busca documentos similares a la consulta"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )

            formatted = []
            if results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i]
                    formatted.append({
                        'text': doc[:800] + "..." if len(doc) > 800 else doc,
                        'metadata': metadata,
                        'score': 1 - (results['distances'][0][i] if results['distances'] else 0),
                        'pdf_title': metadata.get('pdf_title', 'Sin título')
                    })

            return formatted

        except Exception as e:
            print(f"❌ Error en búsqueda: {e}")
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
