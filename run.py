# -*- coding: utf-8 -*-
"""run.py - Punto de entrada para RAG Hispanidad con Gradio"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Añadir src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def setup_colab():
    """Configurar entorno de Google Colab"""
    print("🔧 Configurando entorno de Colab...")
    
    # Montar Google Drive
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    
    # Crear directorios necesarios
    DRIVE_PATH = "/content/drive/MyDrive/RAG_Hispanidad"
    os.makedirs(DRIVE_PATH, exist_ok=True)
    os.makedirs(f"{DRIVE_PATH}/vector_db", exist_ok=True)
    os.makedirs(f"{DRIVE_PATH}/pdf_storage", exist_ok=True)
    
    print(f"📁 Drive montado en: {DRIVE_PATH}")
    return DRIVE_PATH

def install_dependencies():
    """Instalar dependencias necesarias"""
    print("📦 Verificando dependencias...")
    
    # Lista de paquetes necesarios
    packages = [
        "torch==2.3.0",
        "torchvision==0.18.0", 
        "torchaudio==2.3.0",
        "transformers",
        "sentence-transformers",
        "chromadb==0.4.22",
        "pypdf",
        "PyPDF2",
        "pdfplumber",
        "pymupdf",
        "gradio==4.12.0",
        "accelerate",
        "bitsandbytes",
        "sentencepiece",
        "protobuf"
    ]
    
    import subprocess
    import importlib
    
    for package in packages:
        pkg_name = package.split('==')[0].replace('-', '_')
        try:
            importlib.import_module(pkg_name)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   📥 Instalando {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

def main():
    """Función principal"""
    print("\n" + "="*70)
    print("🏛️  SISTEMA RAG HISPANIDAD - CHAT CON PDFS HISTÓRICOS")
    print("="*70)
    
    # 1. Configurar entorno (especialmente para Colab)
    is_colab = 'google.colab' in sys.modules
    if is_colab:
        print("🔍 Detectado Google Colab")
        setup_colab()
    else:
        print("🔍 Entorno local detectado")
    
    # 2. Verificar/instalar dependencias
    install_dependencies()
    
    # 3. Importar componentes del sistema RAG
    print("\n📚 Cargando módulos del sistema...")
    try:
        from src.rag_system import PDFRAGSystem
        from src.gradio_interface import GradioInterface
        print("✅ Módulos cargados exitosamente")
    except ImportError as e:
        print(f"❌ Error importando módulos: {e}")
        print("💡 Asegúrate de que todos los archivos están en src/")
        return
    
    # 4. Inicializar sistema RAG
    print("\n" + "="*60)
    print("🚀 INICIALIZANDO SISTEMA RAG...")
    print("="*60)
    
    try:
        # Inicializar sistema principal
        rag_system = PDFRAGSystem()
        
        # Crear interfaz Gradio
        interface = GradioInterface(rag_system)
        demo = interface.create_interface()
        
        # 5. Mostrar información del sistema
        stats = rag_system.get_system_info()
        print("\n📊 SISTEMA LISTO:")
        print(f"   • PDFs procesados: {stats.get('total_pdfs', 0)}")
        print(f"   • Chunks indexados: {stats.get('total_chunks', 0):,}")
        print(f"   • GPU activa: {'✅ Sí' if stats.get('gpu', False) else '❌ No'}")
        print(f"   • Modelo: {stats.get('model', 'Desconocido')}")
        
        # 6. Lanzar aplicación
        print("\n" + "="*60)
        print("🌐 LANZANDO INTERFAZ WEB...")
        print("="*60)
        
        print("\n🎯 **Instrucciones:**")
        print("1. Sube PDFs históricos usando el panel izquierdo")
        print("2. Haz clic en '🔧 Procesar PDFs' para indexarlos")
        print("3. Pregunta sobre cualquier tema histórico")
        print("4. ¡Todo se guarda automáticamente en tu Google Drive!")
        
        # Configuración de lanzamiento
        launch_kwargs = {
            'debug': False,
            'share': True,  # Para URL pública
            'server_name': '0.0.0.0',
            'server_port': 7860
        }
        
        # En Colab, ajustar parámetros
        if is_colab:
            print("\n⏳ Generando URL pública (puede tardar unos segundos)...")
            launch_kwargs['share'] = True
            
        # Lanzar aplicación
        demo.launch(**launch_kwargs)
        
    except Exception as e:
        print(f"\n❌ ERROR CRÍTICO: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Posibles soluciones:")
        print("1. Verifica que todas las dependencias están instaladas")
        print("2. Revisa que los archivos en src/ estén completos")
        print("3. Intenta reiniciar el runtime en Colab")

def test_imports():
    """Función de prueba para verificar imports"""
    print("🧪 Probando imports...")
    
    modules = [
        ('config', 'Configuración'),
        ('pdf_extractor', 'SmartPDFExtractor'),
        ('pdf_manager', 'PDFManager'),
        ('vector_store', 'PersistentVectorStore'),
        ('chat_engine', 'ChatEngine'),
        ('rag_system', 'PDFRAGSystem'),
        ('gradio_interface', 'GradioInterface'),
    ]
    
    for module_name, description in modules:
        try:
            module = __import__(f'src.{module_name}', fromlist=[''])
            print(f"   ✅ {module_name}: {description}")
        except ImportError as e:
            print(f"   ❌ {module_name}: {e}")

if __name__ == "__main__":
    # Primero probar imports
    test_imports()
    
    # Ejecutar aplicación
    main()
