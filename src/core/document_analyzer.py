# -*- coding: utf-8 -*-
"""Analizador de documentos que se ejecuta DURANTE la indexación"""

import re
from typing import Dict, List, Tuple
from collections import Counter
import hashlib

class DocumentAnalyzer:
    """Analiza documentos COMPLETOS una sola vez durante la indexación"""
    
    def __init__(self):
        self.cache = {}
        
    def generate_document_id(self, text: str, title: str = "") -> str:
        """Genera ID único para el documento"""
        content_hash = hashlib.md5(text.encode()).hexdigest()
        title_hash = hashlib.md5(title.encode()).hexdigest()[:8]
        return f"doc_{title_hash}_{content_hash[:8]}"
    
    def extract_key_themes(self, text: str, max_themes: int = 10) -> List[str]:
        """Extrae temas principales del documento (ejecutar UNA VEZ)"""
        if len(text) < 100:
            return []
        
        # Limpiar y tokenizar
        words = re.findall(r'\b[a-záéíóúñ]{4,}\b', text.lower())
        
        # Filtrar stopwords en español
        stopwords = {
            'para', 'como', 'este', 'esta', 'esto', 'pero', 'porque', 
            'más', 'muy', 'hay', 'son', 'con', 'del', 'las', 'los'
        }
        
        filtered_words = [w for w in words if w not in stopwords]
        
        # Contar frecuencia
        word_freq = Counter(filtered_words)
        
        # Obtener temas más frecuentes
        themes = [word for word, count in word_freq.most_common(max_themes) 
                 if count >= 3]
        
        return themes[:5]  # Solo top 5
    
    def create_intelligent_summary(self, text: str, max_length: int = 500) -> str:
        """Crea un resumen inteligente del documento (ejecutar UNA VEZ)"""
        if len(text) < 300:
            return text
        
        # 1. Tomar primeras oraciones (introducción)
        sentences = text.split('. ')
        summary_parts = []
        
        if len(sentences) > 0:
            summary_parts.append(sentences[0])
        
        # 2. Buscar oraciones con palabras clave importantes
        important_keywords = ['conclusión', 'resumen', 'importante', 'principal', 
                             'objetivo', 'propósito', 'resultado']
        
        for sentence in sentences[1:min(20, len(sentences))]:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in important_keywords):
                if sentence not in summary_parts:
                    summary_parts.append(sentence)
        
        # 3. Tomar última oración (conclusión)
        if len(sentences) > 2:
            last_sentence = sentences[-1]
            if last_sentence not in summary_parts:
                summary_parts.append(last_sentence)
        
        # 4. Combinar y limitar longitud
        summary = '. '.join(summary_parts[:5]) + '.'
        
        if len(summary) > max_length:
            summary = summary[:max_length] + '...'
        
        return summary
    
    def extract_key_entities(self, text: str) -> Dict[str, List[str]]:
        """Extrae entidades clave: personas, lugares, fechas"""
        entities = {
            'persons': [],
            'locations': [],
            'dates': [],
            'key_phrases': []
        }
        
        # Patrones simples para extracción
        # Personas (nombres propios con mayúscula)
        person_pattern = r'\b[A-Z][a-záéíóúñ]+\s+[A-Z][a-záéíóúñ]+\b'
        persons = re.findall(person_pattern, text)
        entities['persons'] = list(set(persons))[:5]
        
        # Fechas (patrones simples)
        date_pattern = r'\b\d{1,2}\s+de\s+[a-z]+\s+de\s+\d{4}\b'
        dates = re.findall(date_pattern, text.lower())
        entities['dates'] = dates[:5]
        
        # Frases clave (oraciones con verbos importantes)
        key_verbs = ['demuestra', 'confirma', 'establece', 'propone', 'sugiere']
        sentences = text.split('. ')
        for sentence in sentences[:20]:
            if any(verb in sentence.lower() for verb in key_verbs):
                if len(sentence) < 150:
                    entities['key_phrases'].append(sentence.strip())
        
        return entities
    
    def analyze_complete_document(self, text: str, title: str = "") -> Dict:
        """
        Análisis COMPLETO del documento (ejecutar SOLO UNA VEZ por documento)
        
        Returns:
            Dict con toda la metadata enriquecida
        """
        doc_id = self.generate_document_id(text, title)
        
        analysis = {
            'doc_id': doc_id,
            'title': title,
            'total_length': len(text),
            'word_count': len(text.split()),
            'themes': self.extract_key_themes(text),
            'summary': self.create_intelligent_summary(text),
            'entities': self.extract_key_entities(text),
            'chunk_count': 0,  # Se llena después
            'analysis_version': '1.0'
        }
        
        # Cache local para reutilización
        self.cache[doc_id] = analysis
        
        return analysis
    
    def analyze_for_query(self, pre_analyzed_doc: Dict, question_keywords: List[str]) -> Dict:
        """
        Análisis RÁPIDO para una consulta específica (usa análisis previo)
        
        Args:
            pre_analyzed_doc: Análisis completo ya hecho
            question_keywords: Keywords de la pregunta actual
        """
        # INCREMENTAL: Solo calcula relevancia para esta pregunta
        relevance_score = 0
        matching_themes = []
        
        # Verificar coincidencia con temas
        for theme in pre_analyzed_doc.get('themes', []):
            for kw in question_keywords:
                if kw in theme.lower():
                    relevance_score += 2
                    matching_themes.append(theme)
                    break
        
        # Verificar coincidencia con resumen
        summary = pre_analyzed_doc.get('summary', '').lower()
        for kw in question_keywords:
            if kw in summary:
                relevance_score += 1
        
        return {
            'doc_id': pre_analyzed_doc['doc_id'],
            'title': pre_analyzed_doc['title'],
            'relevance_score': min(10, relevance_score),
            'matching_themes': matching_themes[:3],
            'summary': pre_analyzed_doc.get('summary', '')[:200] + '...',
            'pre_analyzed': True  # Flag importante
        }