# -*- coding: utf-8 -*-
"""Interfaz de usuario con Gradio adaptada a la nueva arquitectura"""

import gradio as gr
import torch
from typing import List, Tuple, Dict, Any
import json

class GradioInterface:
    def __init__(self, orchestrator):
        """
        Args:
            orchestrator: Instancia de RAGOrchestrator (nueva arquitectura)
        """
        self.orchestrator = orchestrator
        print("🎨 Interfaz adaptada a arquitectura optimizada")
        print("🤖 Selector de modelos habilitado")

    def create_interface(self):
        """Crea la interfaz de Gradio para la nueva arquitectura"""
        print("\n" + "="*60)
        print("📤 CREANDO INTERFAZ GRADIO OPTIMIZADA CON SELECTOR DE MODELOS")
        print("="*60)

        # DEBUG: Verificar que el orchestrator tiene los métodos necesarios
        print("\n🔍 VERIFICANDO MÉTODOS DEL ORCHESTRATOR:")
        print(f"   get_available_models: {hasattr(self.orchestrator, 'get_available_models')}")
        print(f"   get_current_model_info: {hasattr(self.orchestrator, 'get_current_model_info')}")
        print(f"   change_model: {hasattr(self.orchestrator, 'change_model')}")
        
        # Obtener información inicial de modelos
        try:
            available_models = self.orchestrator.get_available_models()
            current_model = self.orchestrator.get_current_model_info()
            print(f"\n📊 MODELOS DISPONIBLES: {list(available_models.keys())}")
            print(f"📊 MODELO ACTUAL: {current_model}")
        except Exception as e:
            print(f"❌ ERROR obteniendo información de modelos: {e}")
            available_models = {'salamandra7b': {'display_name': 'Salamandra 7B', 'description': 'Modelo por defecto'}}
            current_model = {'key': 'salamandra7b', 'display_name': 'Salamandra 7B'}

        with gr.Blocks(title="RAG Hispanidad - Arquitectura Optimizada", 
                      theme=gr.themes.Soft(), css=".gradio-container {max-width: 1400px !important;}") as demo:
            
            # Estado
            chat_history = gr.State([])
            
            # ===== HEADER =====
            gr.Markdown("# 🏛️ **RAG Hispanidad - Arquitectura Optimizada**")
            gr.Markdown("### 🤖 Chat con PDFs Históricos usando análisis inteligente")
            
            # ===== CHAT PRINCIPAL =====
            with gr.Row():
                with gr.Column(scale=3):
                    # Chatbot
                    chatbot = gr.Chatbot(
                        label="💬 Conversación Inteligente",
                        height=500,
                        bubble_full_width=False
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
                with gr.Column(scale=1, min_width=400):
                    # ===== SELECTOR DE MODELO =====
                    gr.Markdown("### 🤖 **Configuración del Modelo**")
                    
                    with gr.Group():
                        # Selector de modelo
                        model_selector = gr.Dropdown(
                            choices=list(available_models.keys()),
                            value=current_model.get('key', 'salamandra7b'),
                            label="Selecciona el modelo de lenguaje",
                            info="Cambia entre diferentes modelos según tus necesidades"
                        )
                        
                        # Botón para cambiar modelo
                        change_model_btn = gr.Button(
                            "🔄 Cambiar Modelo",
                            variant="primary",
                            size="sm",
                            scale=1
                        )
                        
                        # Mostrar información detallada del modelo seleccionado
                        model_info_text = gr.Textbox(
                            label="📋 Información del modelo seleccionado",
                            value=self._format_model_info(current_model),
                            interactive=False,
                            lines=4
                        )
                        
                        # Estado del cambio de modelo
                        model_change_status = gr.Textbox(
                            label="📢 Estado",
                            value="Selecciona un modelo y haz clic en 'Cambiar Modelo'",
                            interactive=False,
                            lines=2
                        )
                    
                    # Separador
                    gr.Markdown("---")
                    
                    # Gestión de documentos
                    gr.Markdown("### 📄 **Gestión de Documentos**")
                    
                    with gr.Group():
                        pdf_upload = gr.File(
                            label="Arrastra o selecciona PDFs históricos",
                            file_types=[".pdf"],
                            file_count="multiple",
                            height=100
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
                            lines=3
                        )
                    
                    # Separador
                    gr.Markdown("---")
                    
                    # Estadísticas del sistema
                    gr.Markdown("### 📊 **Estado del Sistema**")
                    
                    stats_display = gr.Markdown(
                        value="Cargando estadísticas...",
                        label="Estadísticas en tiempo real"
                    )
                    
                    # Botones de sistema
                    with gr.Row():
                        refresh_btn = gr.Button("🔄 Actualizar", variant="secondary", size="sm")
            
            # ===== INFORMACIÓN DEL SISTEMA =====
            gr.Markdown("---")
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Mostrar información del modelo actual
                    model_display = self._get_current_model_display()
                    gr.Markdown(f"""
                    ### 🏗️ **Arquitectura Optimizada**
                    - **Modelo actual:** {model_display}
                    - **Embeddings:** sentence-transformers/paraphrase-multilingual-mpnet-base-v2
                    - **GPU:** {'✅ Disponible' if torch.cuda.is_available() else '❌ Solo CPU'}
                    - **Análisis:** Durante indexación (1x por documento)
                    """)
                
                with gr.Column(scale=2):
                    gr.Markdown("""
                    ### 🎯 **Cómo funciona:**
                    1. **Selecciona un modelo** → Elige entre Salamandra-2B (rápido), 7B (equilibrado) o ALIA-40B (avanzado)
                    2. **Sube PDFs** → Se analizan completamente una sola vez
                    3. **Haz preguntas** → Búsqueda rápida en chunks ya analizados
                    4. **Obtén respuestas** → Basadas en metadatos + conocimiento general
                    
                    **Ventajas:** ⚡ Más rápido, 🧠 Memoria optimizada, 🤖 Múltiples modelos
                    """)
            
            # ===== CONEXIONES =====
            
            # 1. ACTUALIZAR INFO DEL MODELO AL SELECCIONAR
            model_selector.change(
                fn=self._on_model_selected,
                inputs=[model_selector],
                outputs=[model_info_text]
            )
            
            # 2. CAMBIO DE MODELO
            change_model_btn.click(
                fn=self._change_model,
                inputs=[model_selector],
                outputs=[model_change_status, model_info_text, stats_display]
            ).then(
                fn=self._update_model_display,
                outputs=[]
            )
            
            # 3. CHAT PRINCIPAL
            submit_btn.click(
                fn=self.chat_function,
                inputs=[user_input, chat_history],
                outputs=[chatbot, user_input]
            )
            
            user_input.submit(
                fn=self.chat_function,
                inputs=[user_input, chat_history],
                outputs=[chatbot, user_input]
            )
            
            # 4. PROCESAMIENTO DE PDFs
            pdf_process_btn.click(
                fn=self.process_pdfs_function,
                inputs=[pdf_upload],
                outputs=[pdf_status, stats_display]
            )
            
            # 5. BOTONES DE CONTROL
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
                fn=self._get_system_stats,
                outputs=[stats_display]
            )
            
            # 6. CARGA INICIAL
            demo.load(
                fn=self._get_system_stats,
                outputs=[stats_display]
            )

        return demo

    # ===== FUNCIONES AUXILIARES PARA MODELOS =====
    
    def _format_model_info(self, model_info: Dict[str, Any]) -> str:
        """Formatea la información del modelo para mostrar"""
        if not model_info:
            return "Información del modelo no disponible"
        
        try:
            info = f"📌 {model_info.get('display_name', 'Modelo desconocido')}\n"
            info += f"📝 {model_info.get('description', 'Sin descripción')}\n"
            info += f"💾 Memoria: {model_info.get('memory_required', 'N/A')}\n"
            info += f"🔤 Tokens máx: {model_info.get('max_tokens', 600)}"
            
            if model_info.get('gpu_sufficient') is False:
                info += "\n⚠️  Este modelo puede requerir más memoria GPU de la disponible"
            
            return info
        except:
            return str(model_info)
    
    def _on_model_selected(self, model_key: str) -> str:
        """Cuando se selecciona un modelo en el dropdown"""
        print(f"🔍 Modelo seleccionado: {model_key}")
        
        try:
            models = self.orchestrator.get_available_models()
            if model_key in models:
                model_info = models[model_key]
                return self._format_model_info(model_info)
            else:
                return f"❌ Modelo '{model_key}' no encontrado"
        except Exception as e:
            return f"❌ Error obteniendo información: {str(e)[:100]}"
    
    def _change_model(self, model_key: str) -> Tuple[str, str, str]:
        self.orchestrator.reload_llm(model_key)
        """Función para cambiar el modelo de lenguaje"""
        print(f"\n🔄 INTENTANDO CAMBIAR MODELO A: {model_key}")
        
        try:
            # Intentar cambiar el modelo
            result = self.orchestrator.change_model(model_key)
            
            if result.get('success', False):
                # Obtener nueva información del modelo
                new_model_info = self.orchestrator.get_current_model_info()
                
                status_msg = f"✅ Modelo cambiado exitosamente a: {new_model_info.get('display_name', model_key)}"
                model_info_text = self._format_model_info(new_model_info)
                
                print(f"   ✅ Cambio exitoso: {status_msg}")
                
                return status_msg, model_info_text, self._get_system_stats()
            else:
                error_msg = result.get('error', 'Error desconocido al cambiar modelo')
                current_info = self.orchestrator.get_current_model_info()
                
                print(f"   ❌ Error: {error_msg}")
                
                return f"❌ {error_msg}", self._format_model_info(current_info), self._get_system_stats()
                
        except Exception as e:
            print(f"   ❌ Excepción: {e}")
            current_info = self.orchestrator.get_current_model_info()
            return f"❌ Error: {str(e)[:100]}", self._format_model_info(current_info), self._get_system_stats()
    
    def _get_current_model_display(self) -> str:
        """Obtiene el nombre del modelo actual para mostrar"""
        try:
            model_info = self.orchestrator.get_current_model_info()
            return model_info.get('display_name', 'Salamandra 7B')
        except:
            return "Salamandra 7B"
    
    def _update_model_display(self):
        """Función vacía para actualizar la UI (se llama después del cambio)"""
        return
    
    def _get_system_stats(self) -> str:
        """Obtiene estadísticas del sistema"""
        try:
            stats = self.orchestrator.get_system_info()
            return self._format_stats_detailed(stats)
        except Exception as e:
            return f"❌ Error obteniendo estadísticas: {str(e)[:100]}"
    
    def _format_stats_detailed(self, stats: Dict[str, Any]) -> str:
        """Formatea las estadísticas para mostrar en Markdown"""
        if not stats:
            return "📊 Estadísticas no disponibles"
        
        try:
            # Obtener información del modelo
            model_info = self.orchestrator.get_current_model_info()
            
            md = f"""## 📊 **ESTADO DEL SISTEMA**

### 🤖 MODELO ACTIVO
• **Nombre:** {model_info.get('display_name', 'Desconocido')}
• **Memoria requerida:** {model_info.get('memory_required', 'N/A')}
• **GPU suficiente:** {'✅ Sí' if model_info.get('gpu_sufficient', True) else '⚠️ Puede haber limitaciones'}

### 📚 DOCUMENTOS
• **PDFs procesados:** {stats.get('total_pdfs', 0)}
• **Chunks indexados:** {stats.get('total_chunks', 0):,}
"""
            
            if torch.cuda.is_available():
                md += f"""
### ⚙️ HARDWARE
• **GPU:** {torch.cuda.get_device_name(0)}
• **Memoria GPU:** {torch.cuda.memory_allocated()/1e9:.1f}GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB
"""
            
            return md
            
        except Exception as e:
            return f"❌ Error formateando estadísticas: {str(e)[:100]}"

    # ===== FUNCIONES PRINCIPALES (MANTENER SIN CAMBIOS) =====
    
    def chat_function(self, message: str, history: List, max_chars: int = 2000, num_docs: int = 3):
        """Función principal del chat"""
        print(f"\n{'='*60}")
        print(f"💬 CHAT: '{message[:80]}...'")
        
        try:
            result = self.orchestrator.query(
                question=message,
                max_docs=num_docs
            )
            
            answer = result['answer']
            sources = result['sources']
            
            if sources:
                sources_text = "\n\n📚 **Fuentes consultadas:**\n"
                for i, source in enumerate(sources):
                    has_analysis = "✅" if source.get('has_analysis') else "⚠️"
                    sources_text += f"{i+1}. {has_analysis} {source['title']} (relevancia: {source['score']})\n"
                answer += sources_text
            
            model_info = self.orchestrator.get_current_model_info()
            answer += f"\n\n---\n"
            answer += f"🤖 **Modelo:** {model_info.get('display_name', 'Desconocido')} | 📚 Docs: {result['docs_used']}"
            
            history.append([message, answer])
            print(f"   ✅ Respuesta generada")
            
            return history, ""
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)[:100]}"
            history.append([message, error_msg])
            return history, ""
    
    def process_pdfs_function(self, files):
        """Procesar PDFs"""
        if not files:
            return "❌ No se seleccionaron archivos", self._get_system_stats()
        
        print(f"\n📤 PROCESANDO {len(files)} PDFs...")
        
        results = []
        total_chunks = 0
        
        for i, file in enumerate(files):
            try:
                result = self.orchestrator.process_document(file, file.name)
                
                if result.get('success', False):
                    chunks = result.get('chunks_added', 0)
                    total_chunks += chunks
                    results.append(f"✅ {result.get('filename', 'PDF')}: {chunks} chunks")
                else:
                    results.append(f"❌ {result.get('filename', 'PDF')}: Error")
                    
            except Exception as e:
                results.append(f"❌ {file.name}: Error")
        
        if total_chunks > 0:
            summary = f"✅ **{len(files)} PDFs procesados** ({total_chunks} chunks)"
        else:
            summary = f"⚠️ {len(files)} PDFs procesados, 0 chunks añadidos"
        
        result_text = f"{summary}\n\nDetalles:\n" + "\n".join(results[:5])
        
        if len(results) > 5:
            result_text += f"\n... y {len(results) - 5} más"
        
        return result_text, self._get_system_stats()
    
    def test_system_function(self, message: str, history: List):
        """Función de prueba"""
        try:
            stats = self.orchestrator.get_system_info()
            model_info = self.orchestrator.get_current_model_info()
            
            test_response = f"""🧪 **Prueba del sistema completada**

✅ **COMPONENTES:**
• **Modelo:** {model_info.get('display_name', 'Desconocido')}
• **PDFs procesados:** {stats.get('total_pdfs', 0)}
• **Chunks indexados:** {stats.get('total_chunks', 0):,}
• **GPU:** {'✅ Disponible' if torch.cuda.is_available() else '❌ Solo CPU'}

💡 **Sistema listo para uso.**"""
            
            history.append([message or "Prueba", test_response])
            return history, ""
            
        except Exception as e:
            error_msg = f"❌ Error en prueba: {str(e)[:100]}"
            history.append([message or "Prueba", error_msg])
            return history, ""