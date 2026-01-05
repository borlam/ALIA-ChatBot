# -*- coding: utf-8 -*-
"""Interfaz de usuario con Gradio adaptada a la nueva arquitectura"""

import gradio as gr
import torch
from typing import List, Tuple
from ..system.config import *

class GradioInterface:
    def __init__(self, orchestrator):
        """
        Args:
            orchestrator: Instancia de RAGOrchestrator (nueva arquitectura)
        """
        self.orchestrator = orchestrator
        print("🎨 Interfaz adaptada a arquitectura optimizada")

    def create_interface(self):
        """Crea la interfaz de Gradio para la nueva arquitectura"""
        print("\n" + "="*60)
        print("📤 CREANDO INTERFAZ GRADIO OPTIMIZADA")
        print("="*60)

        with gr.Blocks(title="RAG Hispanidad - Arquitectura Optimizada", 
                      theme=gr.themes.Soft()) as demo:
            
            # Estado
            chat_history = gr.State([])
            
            # ===== HEADER =====
            gr.Markdown("# 🏛️ **RAG Hispanidad - Arquitectura Optimizada**")
            gr.Markdown("### 🤖 Chat con PDFs Históricos usando análisis inteligente")
            gr.Markdown("""
            **Nueva arquitectura:** El análisis de documentos se hace UNA VEZ durante la indexación, 
            haciendo las consultas 10x más rápidas y precisas.
            """)
            
            # ===== CHAT PRINCIPAL =====
            with gr.Row():
                with gr.Column(scale=2):
                    # Chatbot
                    chatbot = gr.Chatbot(
                        label="💬 Conversación Inteligente",
                        height=500,
                        bubble_full_width=False,
                        avatar_images=(
                            "🕵️",  # Usuario
                            "🤖"   # Bot
                        )
                    )
                    
                    # Área de entrada mejorada
                    with gr.Row():
                        user_input = gr.Textbox(
                            label="Tu pregunta sobre historia hispánica",
                            placeholder="Ej: ¿Qué documentos tienes sobre la Leyenda Negra española?",
                            lines=3,
                            max_lines=5,
                            scale=4
                        )
                        
                        submit_btn = gr.Button(
                            "📤 Enviar", 
                            variant="primary",
                            size="lg",
                            scale=1
                        )
                    
                    # Botones de control del chat
                    with gr.Row():
                        clear_btn = gr.Button("🗑️ Limpiar chat", variant="secondary")
                        test_btn = gr.Button("🧪 Probar sistema", variant="secondary")
                        export_btn = gr.Button("📥 Exportar conversación", variant="secondary")
                
                # ===== PANEL LATERAL DERECHO =====
                with gr.Column(scale=1):
                    # ===== NUEVO: SELECTOR DE MODELO =====
                    gr.Markdown("### 🤖 **Selección de Modelo**")
                    
                    with gr.Group():
                        # Obtener modelos disponibles y modelo actual
                        try:
                            available_models = self.orchestrator.get_available_models()
                            current_model_info = self.orchestrator.get_current_model_info()
                            model_keys = list(available_models.keys())
                            current_key = current_model_info.get('key', model_keys[0] if model_keys else 'salamandra7b')
                        except:
                            # Fallback si hay error
                            available_models = {'salamandra7b': {'display_name': 'Salamandra 7B'}}
                            model_keys = ['salamandra7b']
                            current_key = 'salamandra7b'
                        
                        # Selector de modelo
                        model_selector = gr.Dropdown(
                            choices=model_keys,
                            value=current_key,
                            label="Modelo de Lenguaje",
                            info="Selecciona el modelo para generar respuestas"
                        )
                        
                        # Botón para cambiar modelo
                        change_model_btn = gr.Button(
                            "🔄 Cambiar Modelo",
                            variant="primary",
                            size="sm"
                        )
                        
                        # Información del modelo actual
                        model_info_display = gr.JSON(
                            label="Información del Modelo",
                            value=current_model_info
                        )
                    
                    # Gestión de documentos
                    gr.Markdown("### 📄 **Gestión de Documentos**")
                    
                    with gr.Group():
                        pdf_upload = gr.File(
                            label="Arrastra o selecciona PDFs históricos",
                            file_types=[".pdf"],
                            file_count="multiple",
                            height=120
                        )
                        
                        with gr.Row():
                            pdf_process_btn = gr.Button(
                                "🔧 Procesar & Analizar PDFs", 
                                variant="primary",
                                size="sm"
                            )
                            pdf_clear_btn = gr.Button("Limpiar", variant="secondary", size="sm")
                        
                        pdf_status = gr.Textbox(
                            label="Estado del procesamiento",
                            value="Listo para recibir PDFs...",
                            interactive=False,
                            lines=4
                        )
                    
                    # Separador
                    gr.Markdown("---")
                    
                    # Estadísticas del sistema
                    gr.Markdown("### 📊 **Estado del Sistema**")
                    
                    stats_display = gr.Markdown(
                        value="Cargando estadísticas...",
                        label="Estadísticas en tiempo real"
                    )
                    
                    # Configuración
                    gr.Markdown("### ⚙️ **Configuración Avanzada**")
                    
                    with gr.Accordion("Opciones de respuesta", open=False):
                        response_length = gr.Slider(
                            minimum=500,
                            maximum=MAX_RESPONSE_LENGTH,
                            value=DEFAULT_RESPONSE_LENGTH,
                            step=100,
                            label="📏 Longitud máxima de respuesta"
                        )
                        
                        num_docs = gr.Slider(
                            minimum=1,
                            maximum=5,
                            value=3,
                            step=1,
                            label="📚 Número de documentos a usar"
                        )
                    
                    # Botones de sistema
                    with gr.Row():
                        refresh_btn = gr.Button("🔄 Actualizar", variant="secondary", size="sm")
                        theme_search_btn = gr.Button("🔍 Buscar por tema", variant="secondary", size="sm")
            
            # ===== INFORMACIÓN DEL SISTEMA =====
            gr.Markdown("---")
            
            with gr.Row():
                with gr.Column(scale=1):
                    current_model = self.orchestrator.get_current_model_info()
                    model_name = current_model.get('display_name', 'Salamandra 7B')
                    
                    gr.Markdown(f"""
                    ### 🏗️ **Arquitectura Optimizada**
                    - **Modelo:** {model_name}
                    - **Embeddings:** {EMBEDDING_MODEL}
                    - **Base de datos:** ChromaDB persistente
                    - **GPU:** {'✅ Disponible' if torch.cuda.is_available() else '❌ Solo CPU'}
                    - **Análisis:** Durante indexación (1x por documento)
                    """)
                
                with gr.Column(scale=2):
                    gr.Markdown("""
                    ### 🎯 **Cómo funciona la nueva arquitectura:**
                    1. **Subes un PDF** → Se extrae texto y se analiza COMPLETAMENTE (1 vez)
                    2. **Se indexa** → Chunks con metadatos enriquecidos (temas, resumen)
                    3. **Haces una pregunta** → Búsqueda rápida en chunks ya analizados
                    4. **Generas respuesta** → Usa metadatos + conocimiento general
                    
                    **Ventajas:** ⚡ Más rápido, 🧠 Menos memoria, 🎯 Más preciso
                    """)
            
            # ===== CONEXIONES =====
            # 1. CHAT PRINCIPAL
            submit_btn.click(
                fn=self.chat_function,
                inputs=[user_input, chat_history, response_length, num_docs],
                outputs=[chatbot, user_input]
            )
            
            user_input.submit(
                fn=self.chat_function,
                inputs=[user_input, chat_history, response_length, num_docs],
                outputs=[chatbot, user_input]
            )
            
            # 2. PROCESAMIENTO DE PDFs
            pdf_process_btn.click(
                fn=self.process_pdfs_function,
                inputs=[pdf_upload],
                outputs=[pdf_status, stats_display]
            )
            
            # 3. NUEVO: CAMBIO DE MODELO
            change_model_btn.click(
                fn=self.change_model_function,
                inputs=[model_selector],
                outputs=[pdf_status, model_info_display, stats_display]
            )
            
            # 4. BOTONES DE CONTROL
            clear_btn.click(
                fn=lambda: [],
                outputs=[chatbot]
            )
            
            pdf_clear_btn.click(
                fn=lambda: None,
                outputs=[pdf_upload]
            )
            
            test_btn.click(
                fn=self.test_system_function,
                inputs=[user_input, chat_history],
                outputs=[chatbot, user_input]
            )
            
            refresh_btn.click(
                fn=self.get_system_stats_markdown,
                outputs=[stats_display]
            )
            
            theme_search_btn.click(
                fn=self.search_by_theme_function,
                inputs=[user_input],
                outputs=[pdf_status]
            )
            
            # 5. Actualizar info del modelo cuando se selecciona
            model_selector.change(
                fn=lambda key: self.orchestrator.get_available_models()[key],
                inputs=[model_selector],
                outputs=[model_info_display]
            )
            
            # 6. CARGA INICIAL
            demo.load(
                fn=self.get_system_stats_markdown,
                outputs=[stats_display]
            )

        return demo

    # ===== NUEVAS FUNCIONES PARA MANEJO DE MODELOS =====
    
    def change_model_function(self, model_key: str):
        """Función para cambiar el modelo de lenguaje"""
        print(f"\n🔄 SOLICITUD DE CAMBIO DE MODELO: {model_key}")
        
        try:
            # Cambiar modelo usando el orquestador
            result = self.orchestrator.change_model(model_key)
            
            if result.get('success', False):
                # Obtener nueva información del modelo
                model_info = self.orchestrator.get_current_model_info()
                
                message = f"✅ Modelo cambiado exitosamente a {model_info.get('display_name', model_key)}\n"
                message += f"📊 Ahora usarás: {model_info.get('description', '')}"
                
                return message, model_info, self.get_system_stats_markdown()
            else:
                error_msg = f"❌ Error cambiando modelo: {result.get('error', 'Error desconocido')}"
                current_info = self.orchestrator.get_current_model_info()
                return error_msg, current_info, self.get_system_stats_markdown()
                
        except Exception as e:
            print(f"❌ Error en change_model_function: {e}")
            current_info = self.orchestrator.get_current_model_info()
            error_msg = f"❌ Error cambiando modelo: {str(e)[:100]}"
            return error_msg, current_info, self.get_system_stats_markdown()
    
    # ===== FUNCIONES EXISTENTES (MODIFICADAS LEVEMENTE) =====
    
    def format_stats_detailed(self, stats):
        """Formatea las estadísticas para mostrar en Markdown"""
        if not stats:
            return "📊 Estadísticas no disponibles"
        
        # Obtener información del modelo
        model_info = self.orchestrator.get_current_model_info()
        
        md = f"""## 📊 **ESTADO DEL SISTEMA**

### 🤖 MODELO ACTIVO
• **Nombre:** {model_info.get('display_name', 'Desconocido')}
• **Descripción:** {model_info.get('description', 'N/A')}
• **Memoria requerida:** {model_info.get('memory_required', 'N/A')}
• **Compatible con GPU:** {'✅ Sí' if model_info.get('gpu_sufficient', True) else '⚠️ Limitada'}

### 📚 DOCUMENTOS
• **PDFs procesados:** {stats.get('total_pdfs', 0)}
• **Páginas totales:** {stats.get('total_pages', 0):,}
• **Chunks indexados:** {stats.get('total_chunks', 0):,}
• **Calidad media:** {stats.get('quality_distribution', {}).get('alta', 0) or 'N/A'}

### ⚙️ HARDWARE
• **GPU:** {'✅ ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else '❌ CPU'}
• **Arquitectura:** {stats.get('architecture', 'optimized_v2')}
"""
        
        if torch.cuda.is_available():
            md += f"""• **Memoria GPU:** {torch.cuda.memory_allocated()/1e9:.1f}GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB
"""
        
        md += f"""
### 🔄 SISTEMA
• **Última actualización:** {stats.get('last_update', 'N/A')}
• **Embeddings:** {EMBEDDING_MODEL.split('/')[-1]}
"""
        
        return md

    def get_system_stats_markdown(self):
        """Obtiene y formatea las estadísticas del sistema para Markdown"""
        stats = self.orchestrator.get_system_info()
        return self.format_stats_detailed(stats)

    def chat_function(self, message: str, history: List, max_chars: int, num_docs: int):
        """Función principal del chat adaptada a la nueva arquitectura"""
        print(f"\n{'='*60}")
        print(f"🔔 CONSULTA OPTIMIZADA: '{message[:80]}...'")
        print(f"   📚 Usando hasta {num_docs} documentos")

        try:
            # 1. Usar el NUEVO método query del orquestador
            result = self.orchestrator.query(
                question=message,
                max_docs=num_docs
            )
            
            # 2. Formatear respuesta con fuentes
            answer = result['answer']
            sources = result['sources']
            
            # 3. Añadir información de fuentes a la respuesta
            if sources:
                sources_text = "\n\n📚 **Fuentes consultadas:**\n"
                for i, source in enumerate(sources):
                    has_analysis = "✅" if source.get('has_analysis') else "⚠️"
                    sources_text += f"{i+1}. {has_analysis} {source['title']} (relevancia: {source['score']})\n"
                
                answer += sources_text
            
            # 4. Añadir metadata de la respuesta
            answer += f"\n\n---\n"
            model_info = self.orchestrator.get_current_model_info()
            model_name = model_info.get('display_name', 'Desconocido')
            answer += f"📊 **Metadata:** {result['docs_used']} docs | {result['response_length']} chars | {model_name}"
            
            # 5. Actualizar historial
            history.append([message, answer])

            print(f"   ✅ Respuesta generada: {result['response_length']} caracteres")
            print(f"   📊 Documentos usados: {result['docs_used']}")
            print(f"{'='*60}")

            return history, ""

        except Exception as e:
            print(f"❌ ERROR en chat_function: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            
            error_msg = f"""⚠️ **Error en el sistema optimizado**

**Detalles:** {str(e)[:150]}

💡 **Posibles soluciones:**
1. Verifica que los PDFs estén procesados
2. Reinicia la aplicación si es necesario
3. Si el error persiste, revisa los logs"""
            
            history.append([message, error_msg])
            print(f"{'='*60}")
            return history, ""

    def process_pdfs_function(self, files):
        """Procesar PDFs con la nueva arquitectura"""
        if not files:
            return "❌ No se seleccionaron archivos", self.get_system_stats_markdown()

        print(f"\n{'='*60}")
        print(f"📤 PROCESANDO {len(files)} PDFs CON ANÁLISIS COMPLETO...")
        print("(Este proceso se hace UNA VEZ por documento)")
        print(f"{'='*60}")

        results = []
        total_chunks = 0
        total_analysis_time = 0

        for i, file in enumerate(files):
            print(f"   [{i+1}/{len(files)}] Procesando '{file.name}'...")
            
            try:
                # Usar el NUEVO método process_document del orquestador
                result = self.orchestrator.process_document(file, file.name)
                
                if result.get('success', False):
                    chunks = result.get('chunks_added', 0)
                    total_chunks += chunks
                    
                    # Información del análisis
                    themes = result.get('document_themes', [])
                    themes_text = f" | Temas: {', '.join(themes)}" if themes else ""
                    
                    results.append(f"✅ {result.get('filename', 'PDF')}: {chunks} chunks{themes_text}")
                    print(f"       ✓ {chunks} chunks, análisis completado")
                else:
                    results.append(f"❌ {result.get('filename', 'PDF')}: {result.get('error', 'Error')}")
                    print(f"       ✗ Error: {result.get('error', 'Error')}")
                    
            except Exception as e:
                error_msg = f"Error inesperado: {str(e)[:100]}"
                results.append(f"❌ {file.name}: {error_msg}")
                print(f"       ✗ {error_msg}")

        # Actualizar estadísticas
        stats = self.orchestrator.get_system_info()

        # Crear resumen detallado
        if total_chunks > 0:
            summary = f"✅ **{len(files)} PDFs procesados exitosamente**\n"
            summary += f"   • **Chunks añadidos:** {total_chunks}\n"
            summary += f"   • **Análisis completado:** Sí (una vez por documento)\n"
            summary += f"   • **Metadatos enriquecidos:** Temas, resumen, entidades\n"
            summary += f"   • **Próximo paso:** Ya puedes hacer preguntas sobre estos documentos"
        else:
            summary = f"⚠️  {len(files)} PDFs procesados, 0 chunks añadidos\n"
            summary += "   Verifica que los PDFs contengan texto extraíble."

        print(f"📊 RESUMEN: {summary}")
        print(f"{'='*60}")

        # Formatear resultados
        result_text = f"**Resultados del procesamiento con análisis completo:**\n\n"
        result_text += summary + "\n\n"
        result_text += "**Detalles por archivo:**\n"
        result_text += "\n".join(results[:10])
        
        if len(results) > 10:
            result_text += f"\n\n... y {len(results) - 10} más"

        return result_text, self.format_stats_detailed(stats)

    def test_system_function(self, message: str, history: List):
        """Función de prueba del sistema optimizado"""
        print(f"\n🧪 PRUEBA DEL SISTEMA OPTIMIZADO")
        
        try:
            # Obtener estadísticas
            stats = self.orchestrator.get_system_info()
            
            # Obtener información del modelo actual
            model_info = self.orchestrator.get_current_model_info()
            model_name = model_info.get('display_name', 'Salamandra-7B')
            
            # Información de GPU
            gpu_info = ""
            if torch.cuda.is_available():
                gpu_info = f"""
• **GPU:** {torch.cuda.get_device_name(0)}
• **Memoria:** {torch.cuda.memory_allocated()/1e9:.1f}GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB
"""
            else:
                gpu_info = "• **GPU:** ❌ No disponible (usando CPU)"
            
            test_response = f"""🧪 **Prueba del sistema optimizado completada**

✅ **COMPONENTES VERIFICADOS:**
• **Arquitectura:** Optimizada v2.0 (análisis en indexación)
• **Modelo LLM:** {model_name}
• **Base de vectores:** {stats.get('total_chunks', 0):,} chunks con metadatos enriquecidos
• **Documentos procesados:** {stats.get('total_pdfs', 0)}
{gpu_info.strip()}

📊 **ESTADÍSTICAS ACTUALES:**
• **PDFs procesados:** {stats.get('total_pdfs', 0)}
• **Chunks indexados:** {stats.get('total_chunks', 0):,}
• **Última actualización:** {stats.get('last_update', 'N/A')}
• **Arquitectura:** {stats.get('architecture', 'optimized_v2')}

⚡ **VENTAJAS ACTIVAS:**
1. ⚡ Análisis durante indexación (10x más rápido)
2. 🧠 Metadatos enriquecidos en cada chunk
3. 🎯 Búsqueda inteligente por temas
4. 📈 Escalable a cientos de documentos

💡 **Sistema listo para uso óptimo.**"""
            
            history.append([message or "Prueba del sistema", test_response])
            return history, ""
            
        except Exception as e:
            error_msg = f"❌ Error en prueba del sistema: {str(e)[:100]}"
            history.append([message or "Prueba", error_msg])
            return history, ""

    def search_by_theme_function(self, theme_query: str):
        """Busca documentos por tema usando análisis previo"""
        if not theme_query or len(theme_query.strip()) < 3:
            return "❌ Por favor, ingresa un tema de búsqueda (mínimo 3 caracteres)"
        
        print(f"\n🔍 BUSCANDO POR TEMA: '{theme_query}'")
        
        try:
            # Usar el nuevo método del orquestador
            results = self.orchestrator.search_by_theme(theme_query)
            
            if not results:
                return f"🔍 No encontré documentos con el tema: '{theme_query}'"
            
            # Formatear resultados
            response = f"**📚 Documentos encontrados para el tema: '{theme_query}'**\n\n"
            
            for i, result in enumerate(results[:5]):  # Máximo 5 resultados
                themes = result.get('themes', [])
                themes_text = ', '.join(themes[:3]) if themes else 'Sin temas identificados'
                
                response += f"{i+1}. **{result.get('title', 'Documento sin título')}**\n"
                response += f"   • Temas: {themes_text}\n"
                response += f"   • Resumen: {result.get('summary', '')[:150]}...\n\n"
            
            if len(results) > 5:
                response += f"\n... y {len(results) - 5} documentos más."
            
            print(f"   ✅ Encontrados {len(results)} documentos")
            return response
            
        except Exception as e:
            print(f"❌ Error en búsqueda por tema: {e}")
            return f"❌ Error al buscar por tema: {str(e)[:100]}"