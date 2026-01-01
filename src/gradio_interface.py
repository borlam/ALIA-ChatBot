# -*- coding: utf-8 -*-
"""Interfaz de usuario con Gradio"""

import gradio as gr
import torch
from typing import List
from .rag_system import PDFRAGSystem
from .config import *

class GradioInterface:
    def __init__(self, rag_system: PDFRAGSystem):
        self.rag = rag_system

    def create_interface(self):
        """Crea la interfaz de Gradio"""
        print("\n" + "="*60)
        print("📤 CREANDO INTERFAZ GRADIO...")
        print("="*60)

        with gr.Blocks(title="RAG Hispanidad", theme=gr.themes.Soft()) as demo:
            # Estado
            chat_history = gr.State([])

            # Título
            gr.Markdown("## 🤖 **RAG Hispanidad - Chat con PDFs Históricos**")
            gr.Markdown("Sube PDFs históricos y conversa con ellos usando inteligencia artificial")

            # Chatbot
            chatbot = gr.Chatbot(
                label="💬 Conversación",
                height=450,
                bubble_full_width=False
            )

            # Área de entrada
            user_input = gr.Textbox(
                label="Tu pregunta sobre historia hispánica",
                placeholder="Ej: ¿Qué documentos tienes sobre la Leyenda Negra española?",
                lines=3,
                max_lines=5
            )

            # Botones principales
            with gr.Row():
                submit_btn = gr.Button("📤 Enviar pregunta", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ Limpiar chat", variant="secondary")
                test_btn = gr.Button("🧪 Probar sistema", variant="secondary")

            # Panel izquierdo: Gestión de documentos
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📄 **Gestión de PDFs**")

                    pdf_upload = gr.File(
                        label="Arrastra o selecciona PDFs históricos",
                        file_types=[".pdf"],
                        file_count="multiple",
                        height=100
                    )

                    with gr.Row():
                        pdf_process_btn = gr.Button("🔧 Procesar PDFs", variant="primary")
                        pdf_clear_btn = gr.Button("Limpiar", variant="secondary")

                    pdf_status = gr.Textbox(
                        label="Estado de procesamiento",
                        value="Listo para recibir PDFs...",
                        interactive=False,
                        lines=3
                    )

                # Panel derecho: Sistema y estadísticas
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 **Estado del Sistema**")

                    stats_display = gr.Textbox(
                        label="Estadísticas en tiempo real",
                        value="Calculando...",
                        interactive=False,
                        lines=6
                    )

                    gr.Markdown("### ⚙️ **Configuración**")

                    response_length = gr.Slider(
                        minimum=500,
                        maximum=MAX_RESPONSE_LENGTH,
                        value=DEFAULT_RESPONSE_LENGTH,
                        step=100,
                        label="📏 Longitud de respuesta"
                    )

                    refresh_btn = gr.Button("🔄 Actualizar estadísticas", variant="secondary")

            # Información del sistema
            gr.Markdown("---")
            gr.Markdown(f"""
            ### 🏛️ **Sistema RAG Hispanidad**
            - **Modelo:** {MODEL_NAME}
            - **Embeddings:** {EMBEDDING_MODEL}
            - **Base de datos:** ChromaDB persistente en Google Drive
            - **GPU:** {'✅ Disponible' if torch.cuda.is_available() else '❌ Solo CPU'}
            """)

            # ===== CONEXIONES =====
            # 1. CHAT PRINCIPAL
            submit_btn.click(
                fn=self.chat_function,
                inputs=[user_input, chat_history, response_length],
                outputs=[chatbot, user_input]
            )

            user_input.submit(
                fn=self.chat_function,
                inputs=[user_input, chat_history, response_length],
                outputs=[chatbot, user_input]
            )

            # 2. BOTÓN DE PRUEBA
            test_btn.click(
                fn=self.test_system_function,
                inputs=[user_input, chat_history],
                outputs=[chatbot, user_input]
            )

            # 3. PROCESAMIENTO DE PDFs
            pdf_process_btn.click(
                fn=self.process_pdfs_function,
                inputs=[pdf_upload],
                outputs=[pdf_status, stats_display]
            )

            # 4. BOTONES DE LIMPIEZA
            clear_btn.click(
                fn=lambda: [],
                outputs=[chatbot]
            )

            pdf_clear_btn.click(
                fn=lambda: None,
                outputs=[pdf_upload]
            )

            # 5. ACTUALIZACIÓN DE ESTADÍSTICAS
            refresh_btn.click(
                fn=self.get_system_stats,
                outputs=[stats_display]
            )

            # 6. CARGA INICIAL
            demo.load(
                fn=self.get_system_stats,
                outputs=[stats_display]
            )

        return demo

    def chat_function(self, message: str, history: List, max_chars: int):
        """Función principal del chat"""
        print(f"\n{'='*60}")
        print(f"🔔 CHAT LLAMADO: '{message[:80]}...'")

        try:
            # 1. Buscar documentos relevantes
            print("   🔍 Buscando en documentos...")
            docs = self.rag.search_documents(message, n_results=3)
            print(f"   📚 Documentos encontrados: {len(docs)}")

            # 2. Generar respuesta
            if docs:
                print(f"   🤖 Generando respuesta ({max_chars} caracteres máx)...")
                response = self.rag.generate_response(message, docs, max_chars)
                print(f"   ✅ Respuesta generada: {len(response):,} caracteres")
            else:
                response = """📭 **No encontré documentos relevantes en la base de datos.**

💡 **Sugerencias:**
1. Sube PDFs históricos usando el panel izquierdo
2. Haz clic en "Procesar PDFs" para indexarlos
3. Reformula tu pregunta usando términos históricos"""
                print("   ⚠️  No se encontraron documentos")

            # 3. Actualizar historial
            history.append([message, response])

            print(f"   📊 Historial actualizado: {len(history)} intercambios")
            print(f"{'='*60}")

            return history, ""

        except Exception as e:
            print(f"❌ ERROR en chat_function: {type(e).__name__}: {e}")
            error_msg = f"""⚠️ **Error en el sistema**

**Detalles:** {str(e)[:150]}"""
            history.append([message, error_msg])
            print(f"{'='*60}")
            return history, ""

    def process_pdfs_function(self, files):
        """Procesar PDFs - maneja múltiples archivos"""
        if not files:
            return "❌ No se seleccionaron archivos", self.get_system_stats()

        print(f"\n{'='*60}")
        print(f"📤 PROCESANDO {len(files)} PDFs...")

        results = []
        total_chunks = 0

        for i, file in enumerate(files):
            print(f"   [{i+1}/{len(files)}] Procesando...")
            result = self.rag.upload_and_process_pdf(file)

            if result.get('success', False):
                chunks = result.get('chunks_added', 0)
                total_chunks += chunks
                results.append(f"✅ {result.get('filename', 'PDF')}: {chunks} chunks")
            else:
                results.append(f"❌ {result.get('filename', 'PDF')}: {result.get('error', 'Error')}")

        # Actualizar estadísticas
        self.rag.update_stats()
        stats = self.rag.get_system_info()

        # Crear resumen
        if total_chunks > 0:
            summary = f"✅ {len(files)} PDFs procesados, {total_chunks} chunks añadidos"
        else:
            summary = f"⚠️  {len(files)} PDFs procesados, 0 chunks añadidos"

        print(f"📊 RESUMEN: {summary}")
        print(f"{'='*60}")

        result_text = f"**Resultados del procesamiento:**\n\n" + "\n".join(results[:10])
        if len(results) > 10:
            result_text += f"\n\n... y {len(results) - 10} más"

        return result_text, self.format_stats_detailed(stats)

    def test_system_function(self, message: str, history: List):
        """Función de prueba del sistema"""
        print(f"\n🧪 PRUEBA DEL SISTEMA: '{message}'")

        test_response = f"""🧪 **Prueba del sistema completada**

✅ **Componentes verificados:**
• Modelo salamandra-2b: {'🟢 Operativo' if self.rag.chat_engine.model else '🔴 No disponible'}
• Base de vectores: {self.rag.vector_store.get_stats().get('total_chunks', 0):,} chunks
• Embeddings: {'🟢 Operativo' if self.rag.chat_engine.embedder else '🔴 No disponible'}
• GPU: {'🟢 Disponible' if torch.cuda.is_available() else '🟡 Solo CPU'}

📊 **Estadísticas actuales:**
{self.rag.get_system_info().get('total_pdfs', 0)} PDFs procesados
{self
