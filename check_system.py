import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("🔍 DIAGNÓSTICO DEL SISTEMA")
print("="*60)

# Verificar qué módulos existen
print("📁 Archivos en src/system:")
for file in os.listdir("src/system"):
    print(f"  - {file}")

print("\n📦 Importando config...")
try:
    from src.system.config import *
    print("✅ Config importada")
    
    # Verificar si tiene las funciones nuevas
    print("\n🔧 Funciones de configuración:")
    print(f"  get_available_models_list: {'✅' if 'get_available_models_list' in dir() else '❌'}")
    print(f"  set_active_model: {'✅' if 'set_active_model' in dir() else '❌'}")
    print(f"  AVAILABLE_MODELS: {'✅' if 'AVAILABLE_MODELS' in dir() else '❌'}")
    
except ImportError as e:
    print(f"❌ Error importando config: {e}")

print("\n🤖 Intentando importar RAGOrchestrator...")
try:
    from src.system.rag_orchestrator import RAGOrchestrator
    print("✅ RAGOrchestrator importado")
    
    # Crear instancia y verificar métodos
    print("\n🧪 Creando instancia...")
    orchestrator = RAGOrchestrator()
    
    print("\n📋 Métodos disponibles:")
    methods = [m for m in dir(orchestrator) if not m.startswith('_')]
    for method in sorted(methods):
        print(f"  • {method}")
        
    # Verificar métodos específicos
    print("\n🔍 Verificando métodos de modelo:")
    print(f"  get_available_models: {'✅' if hasattr(orchestrator, 'get_available_models') else '❌'}")
    print(f"  get_current_model_info: {'✅' if hasattr(orchestrator, 'get_current_model_info') else '❌'}")
    print(f"  change_model: {'✅' if hasattr(orchestrator, 'change_model') else '❌'}")
    
except ImportError as e:
    print(f"❌ Error importando RAGOrchestrator: {e}")
    import traceback
    traceback.print_exc()