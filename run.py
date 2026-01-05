# -*- coding: utf-8 -*-
"""run.py - Punto de entrada para arquitectura optimizada"""

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
            
        # Crear directorios necesarios en Drive
        directories = [
            DRIVE_PATH,
            os.path.join(DRIVE_PATH, "vector_db"),
            os.path.join(DRIVE_PATH, "pdf_storage"),
            os.path.join(DRIVE_PATH, "cache")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"   📂 {directory}")
        
        print(f"   ✅ Directorios creados en Google Drive")
        return IS_COLAB, DRIVE_PATH
        
    else:
        # En local, usar directorio local
        DRIVE_PATH = os.path.expanduser("~/RAG_Hispanidad")
        print(f"   📁 Usando directorio local: {DRIVE_PATH}")
        
        # Crear directorios necesarios localmente
        directories = [
            DRIVE_PATH,
            os.path.join(DRIVE_PATH, "vector_db"),
            os.path.join(DRIVE_PATH, "pdf_storage"),
            os.path.join(DRIVE_PATH, "cache")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        print(f"   ✅ Directorios creados localmente")
        return IS_COLAB, DRIVE_PATH

def verify_structure():
    """Verifica la estructura de la nueva arquitectura"""
    print("\n🔍 Verificando estructura de archivos...")
    
    required_dirs = [
        'src/core',
        'src/processing', 
        'src/vector',
        'src/llm',
        'src/interface',
        'src/system'
    ]
    
    required_files = [
        'src/core/document_analyzer.py',
        'src/processing/pdf_manager.py',
        'src/vector/vector_store.py',
        'src/llm/chat_engine.py',
        'src/interface/gradio_interface.py',
        'src/system/rag_orchestrator.py',
        'src/system/config.py'
    ]
    
    print("📁 Directorios requeridos:")
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"   ✅ {dir_path}")
        else:
            print(f"   ❌ {dir_path} (FALTANTE)")
    
    print("\n📄 Archivos requeridos:")
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} (FALTANTE)")
    
    # Contar archivos .py en src
    py_files = []
    for root, dirs, files in os.walk('src'):
        for file in files:
            if file.endswith('.py'):
                py_files.append(os.path.join(root, file))
    
    print(f"\n📊 Total archivos Python en src: {len(py_files)}")
    
    return len(py_files) >= 10  # Mínimo 10 archivos para arquitectura completa

def test_critical_imports():
    """Prueba imports críticos de la nueva arquitectura"""
    print("\n🧪 Probando imports críticos...")
    
    modules_to_test = [
        ('src.system.rag_orchestrator', 'RAGOrchestrator'),
        ('src.core.document_analyzer', 'DocumentAnalyzer'),
        ('src.llm.chat_engine', 'ChatEngine'),
        ('src.vector.vector_store', 'PersistentVectorStore'),
        ('src.interface.gradio_interface', 'GradioInterface'),
    ]
    
    all_ok = True
    for module_path, class_name in modules_to_test:
        try:
            # Importar dinámicamente
            import importlib
            module = importlib.import_module(module_path.replace('/', '.'))
            
            # Verificar que la clase existe
            if hasattr(module, class_name):
                print(f"   ✅ {module_path}.{class_name}")
            else:
                print(f"   ❌ {module_path}.{class_name} (clase no encontrada)")
                all_ok = False
                
        except ImportError as e:
            print(f"   ❌ {module_path}: {e}")
            all_ok = False
        except Exception as e:
            print(f"   ⚠️  {module_path}: Error inesperado - {e}")
            all_ok = False
    
    return all_ok

def print_system_info():
    """Imprime información del sistema"""
    print("\n💻 INFORMACIÓN DEL SISTEMA:")
    print(f"   Python: {sys.version.split()[0]}")
    
    try:
        import torch
        print(f"   PyTorch: {torch.__version__}")
        print(f"   CUDA disponible: {'✅ Sí' if torch.cuda.is_available() else '❌ No'}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except ImportError:
        print("   ⚠️  PyTorch no instalado")
    
    try:
        import gradio
        print(f"   Gradio: {gradio.__version__}")
    except ImportError:
        print("   ⚠️  Gradio no instalado")

def show_available_models():
    """Muestra los modelos disponibles y sus características"""
    print("\n🤖 MODELOS DISPONIBLES:")
    print("-" * 50)
    
    try:
        from src.system.config import get_available_models_list, is_gpu_sufficient_for_model
        import torch
        
        models = get_available_models_list()
        
        for key, info in models.items():
            gpu_sufficient = is_gpu_sufficient_for_model(key)
            gpu_icon = "✅" if gpu_sufficient else "⚠️"
            
            print(f"   🔘 {key}:")
            print(f"      📝 Nombre: {info['display_name']}")
            print(f"      📋 Descripción: {info['description']}")
            print(f"      💾 Memoria: {info['memory_required']}")
            print(f"      🎯 Tokens máx: {info['max_tokens']}")
            print(f"      🖥️  GPU: {gpu_icon} {'Suficiente' if gpu_sufficient else 'Puede ser insuficiente'}")
            print()
        
        # Mostrar recomendación
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"💡 RECOMENDACIÓN (GPU: {gpu_memory:.1f}GB):")
            
            if gpu_memory >= 20:
                print("   → Puedes usar ALIA-40B para máxima calidad")
            elif gpu_memory >= 6:
                print("   → Salamandra-7B es la opción óptima")
            else:
                print("   → Salamandra-2B es la mejor opción")
        else:
            print("💡 RECOMENDACIÓN (Solo CPU):")
            print("   → Salamandra-2B es la única opción práctica")
            
    except ImportError as e:
        print(f"⚠️  No se pueden mostrar modelos: {e}")

def main():
    """Función principal"""
    print("\n" + "="*80)
    print("🏛️  SISTEMA RAG HISPANIDAD - ARQUITECTURA OPTIMIZADA v3.0")
    print("🤖 CON SELECCIÓN DE MODELOS: Salamandra-2B/7B o ALIA-40B")
    print("="*80)
    
    # Verificar si se pasa un modelo como argumento
    initial_model = None
    if len(sys.argv) > 1:
        model_arg = sys.argv[1].lower()
        valid_models = ["salamandra2b", "salamandra7b", "alia40b"]
        
        if model_arg in valid_models:
            initial_model = model_arg
            print(f"\n🎯 Modelo inicial solicitado: {model_arg}")
            print(f"   La aplicación iniciará con este modelo")
        else:
            print(f"\n⚠️  Modelo '{model_arg}' no válido.")
            print(f"   Opciones válidas: {', '.join(valid_models)}")
            print("   Iniciando con modelo por defecto (salamandra7b)")
    
    # Mostrar modelos disponibles
    show_available_models()
    
    # 1. Configurar entorno
    print("\n1️⃣ CONFIGURANDO ENTORNO")
    is_colab, data_path = setup_environment()
    
    # 2. Verificar estructura de archivos
    print("\n2️⃣ VERIFICANDO ESTRUCTURA")
    if not verify_structure():
        print("\n⚠️  ADVERTENCIA: Faltan archivos/directorios de la nueva arquitectura")
        print("   La aplicación puede no funcionar correctamente.")
        print("   Continúo con la ejecución, pero puede haber errores.")
    
    # 3. Probar imports críticos
    print("\n3️⃣ PROBANDO IMPORTS CRÍTICOS")
    if not test_critical_imports():
        print("\n❌ ERROR: Faltan módulos críticos")
        print("   Por favor, asegúrate de que todos los archivos de la nueva")
        print("   arquitectura están en sus ubicaciones correctas.")
        return
    
    # 4. Mostrar información del sistema
    print_system_info()
    
    # 5. Importar componentes del nuevo sistema
    print("\n4️⃣ CARGANDO MÓDULOS DEL SISTEMA...")
    try:
        from src.system.rag_orchestrator import RAGOrchestrator
        from src.interface.gradio_interface import GradioInterface
        
        print("✅ Módulos cargados exitosamente")
        print(f"   🏗️  Arquitectura: Optimizada (análisis en indexación)")
        print(f"   📁 Datos: {data_path}")
        
    except ImportError as e:
        print(f"\n❌ ERROR importando módulos: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 SOLUCIÓN: Asegúrate de que:")
        print("   1. Todos los archivos de la nueva arquitectura están en src/")
        print("   2. Los nombres de clases coinciden (RAGOrchestrator, etc.)")
        print("   3. Los imports en los archivos están actualizados")
        return
    
    # 6. Inicializar sistema RAG
    print("\n" + "="*60)
    print("🚀 INICIALIZANDO SISTEMA RAG OPTIMIZADO")
    print("="*60)
    
    try:
        # Inicializar el NUEVO orquestador con modelo inicial si se especificó
        orchestrator = RAGOrchestrator(initial_model_key=initial_model)
        
        # Obtener estadísticas iniciales
        stats = orchestrator.get_system_info()
        
        # 7. Mostrar información del sistema cargado
        print("\n📊 SISTEMA CARGADO EXITOSAMENTE:")
        print(f"   • PDFs procesados: {stats.get('total_pdfs', 0)}")
        print(f"   • Chunks indexados: {stats.get('total_chunks', 0):,}")
        print(f"   • GPU activa: {'✅ Sí' if stats.get('gpu_available', False) else '❌ No'}")
        
        # Mostrar información del modelo
        model_info = stats.get('model', {})
        if isinstance(model_info, dict):
            print(f"   • Modelo activo: {model_info.get('display_name', 'Desconocido')}")
            print(f"   • Descripción: {model_info.get('description', 'N/A')}")
        else:
            print(f"   • Modelo: {stats.get('model', 'Desconocido')}")
        
        print(f"   • Arquitectura: {stats.get('architecture', 'optimized_v2')}")
        
        # 8. Crear interfaz Gradio adaptada
        print("\n5️⃣ CREANDO INTERFAZ WEB...")
        interface = GradioInterface(orchestrator)
        demo = interface.create_interface()
        
        # 9. Lanzar aplicación
        print("\n" + "="*60)
        print("🌐 LANZANDO INTERFAZ WEB")
        print("="*60)
        
        print("\n🎯 **INSTRUCCIONES DE USO:**")
        print("1. 🤖 Selecciona el modelo en el panel derecho (2B, 7B o ALIA-40B)")
        print("2. 📤 Sube PDFs históricos usando el panel izquierdo")
        print("3. 🔧 Haz clic en 'Procesar PDFs' para indexarlos (con análisis completo)")
        print("4. 💬 Pregunta sobre cualquier tema histórico")
        print("5. 📚 Las respuestas usarán análisis previo + conocimiento general")
        print("6. 💾 Todo se guarda automáticamente en Google Drive")
        
        print("\n⚡ **VENTAJAS DE LA NUEVA ARQUITECTURA:**")
        print("   • ⚡ 10x más rápido: Análisis se hace una sola vez")
        print("   • 🧠 Menos memoria: Sin análisis pesado en cada pregunta")
        print("   • 🎯 Más preciso: Metadatos enriquecidos")
        print("   • 🤖 Modelos múltiples: Elige entre 2B, 7B o ALIA-40B")
        print("   • 📈 Escalable: Soporta cientos de PDFs")
        
        # Configuración de lanzamiento
        launch_kwargs = {
            'debug': False,
            'share': is_colab,  # URL pública solo en Colab
            'server_name': '0.0.0.0',
            'server_port': 7860,
            'show_error': True
        }
        
        if is_colab:
            print("\n⏳ Generando URL pública...")
            print("   La URL estará disponible en unos segundos")
            print("   ⚠️  En Colab free, la sesión expira después de un tiempo")
            print("   💡 Usa Ctrl+C para detener y liberar recursos")
        else:
            print(f"\n🌐 Servidor local: http://localhost:7860")
            print("   Presiona Ctrl+C para detener el servidor")
        
        # Instrucciones para cambio de modelo
        print("\n🔄 **CAMBIO DE MODELO DURANTE LA EJECUCIÓN:**")
        print("   • Selecciona un modelo diferente en el panel derecho")
        print("   • Haz clic en '🔄 Cambiar Modelo'")
        print("   • El sistema recargará automáticamente el nuevo modelo")
        print("   • ⚠️ El cambio puede tardar 1-2 minutos dependiendo del modelo")
        
        print("\n" + "="*60)
        print("✅ SISTEMA LISTO - ESPERANDO CONEXIONES...")
        print("="*60)
        
        # Lanzar aplicación
        demo.launch(**launch_kwargs)
        
    except Exception as e:
        print(f"\n❌ ERROR CRÍTICO durante la inicialización: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n🔧 **POSIBLES SOLUCIONES:**")
        print("1. Verifica que todos los archivos de la nueva arquitectura existen")
        print("2. Comprueba que los imports en los archivos son correctos")
        print("3. Asegúrate de que las dependencias están instaladas")
        print("4. Si usas Colab, reinicia el runtime y vuelve a intentar")
        
        # Sugerencia específica para errores comunes
        if "No module named" in str(e):
            print(f"\n💡 ERROR DE IMPORT: {e}")
            print("   Ejecuta: pip install -r requirements.txt")
        elif "CUDA out of memory" in str(e):
            print(f"\n💡 ERROR DE MEMORIA GPU: {e}")
            print("   Usa un modelo más pequeño: python run.py salamandra2b")
            print("   O libera memoria GPU: import torch; torch.cuda.empty_cache()")

if __name__ == "__main__":
    main()