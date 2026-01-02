# -*- coding: utf-8 -*-
"""run.py - Punto de entrada optimizado para Colab y local"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Añadir src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def setup_environment():
    """Configurar entorno según el contexto"""
    print("🔧 Configurando entorno...")
    
    # Verificar si estamos en Colab
    try:
        import google.colab
        IS_COLAB = True
        print("   🖥️  Google Colab detectado")
    except ImportError:
        IS_COLAB = False
        print("   💻 Entorno local detectado")
    
    if IS_COLAB:
        # En Colab, NO usar drive.mount() desde script
        # El usuario debe montar Drive manualmente desde una celda
        DRIVE_PATH = "/content/drive/MyDrive/RAG_Hispanidad"
        print(f"   📁 Usando ruta de Drive: {DRIVE_PATH}")
        
        # Verificar si Drive está montado
        if not os.path.exists("/content/drive"):
            print("\n⚠️  ATENCIÓN: Google Drive no está montado")
            print("   Por favor, ejecuta en una celda:")
            print("   from google.colab import drive")
            print("   drive.mount('/content/drive')")
            print("\n   Luego vuelve a ejecutar: python run.py")
            sys.exit(1)
    else:
        # En local, usar directorio local
        DRIVE_PATH = os.path.expanduser("~/RAG_Hispanidad")
        print(f"   📁 Usando directorio local: {DRIVE_PATH}")
    
    # Crear directorios necesarios
    os.makedirs(DRIVE_PATH, exist_ok=True)
    os.makedirs(os.path.join(DRIVE_PATH, "vector_db"), exist_ok=True)
    os.makedirs(os.path.join(DRIVE_PATH, "pdf_storage"), exist_ok=True)
    
    print(f"   ✅ Directorios creados en: {DRIVE_PATH}")
    return IS_COLAB, DRIVE_PATH

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
            __import__(f'src.{module_name}')
            print(f"   ✅ {module_name}: {description}")
        except ImportError as e:
            print(f"   ❌ {module_name}: {e}")

def main():
    """Función principal"""
    print("\n" + "="*70)
    print("🏛️  SISTEMA RAG HISPANIDAD - CHAT CON PDFS HISTÓRICOS")
    print("="*70)
    
    # 1. Configurar entorno
    is_colab, data_path = setup_environment()
    
    # 2. Actualizar config.py con la ruta correcta
    config_path = os.path.join(os.path.dirname(__file__), 'src', 'config.py')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # Actualizar rutas en config.py
        config_content = config_content.replace(
            'DRIVE_PATH = "/content/drive/MyDrive/RAG_Hispanidad"',
            f'DRIVE_PATH = "{data_path}"'
        )
        
        with open(config_path, 'w') as f:
            f.write(config_content)
        print(f"   📝 Config actualizada con ruta: {data_path}")
    
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
        print("4. ¡Todo se guarda automáticamente!")
        
        # Configuración de lanzamiento
        launch_kwargs = {
            'debug': False,
            'share': is_colab,  # URL pública solo en Colab
            'server_name': '0.0.0.0',
            'server_port': 7860
        }
        
        if is_colab:
            print("\n⏳ Generando URL pública...")
            print("   La URL estará disponible en unos segundos")
        
        # Lanzar aplicación
        demo.launch(**launch_kwargs)
        
    except Exception as e:
        print(f"\n❌ ERROR CRÍTICO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Primero probar imports
    test_imports()
    
    # Ejecutar aplicación
    main()
