# run.py - Punto de entrada único
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import settings
from src.file_manager import DriveManager
from src.pdf_processor import PDFProcessor
from src.chatbot import ChatBot

def main():
    print("🚀 Iniciando ALIA-ChatBot...")
    
    # 1. Gestión de archivos
    fm = DriveManager()
    pdf_files = fm.get_available_pdfs()
    
    if not pdf_files:
        print("No hay PDFs. Subiendo archivos de ejemplo...")
        pdf_files = fm.upload_pdfs()
    
    # 2. Procesar PDFs
    print(f"📄 Procesando {len(pdf_files)} archivos PDF...")
    processor = PDFProcessor()
    vector_store = processor.process_pdfs(pdf_files)
    
    # 3. Inicializar chatbot
    print("🤖 Inicializando modelo ALIA...")
    bot = ChatBot(vector_store)
    bot.initialize_model(api_key=os.getenv("HF_TOKEN"))
    
    # 4. Bucle de conversación
    print("\n" + "="*50)
    print("Chatbot listo! Escribe 'salir' para terminar.")
    print("="*50)
    
    while True:
        try:
            user_input = input("\n👤 Tú: ")
            if user_input.lower() == 'salir':
                print("👋 ¡Hasta luego!")
                break
            
            response, sources = bot.ask(user_input)
            print(f"\n🤖 Bot: {response}")
            
            if settings.DEBUG and sources:
                print("\n📚 Fuentes consultadas:")
                for i, doc in enumerate(sources[:2], 1):
                    print(f"   {i}. {doc.page_content[:150]}...")
                    
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrumpido por el usuario")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()