import sys
import os
import importlib.util

# Aseguramos que Colab encuentre la carpeta /src
sys.path.append("/content/PicToMusic")

# Ruta absoluta al archivo pipeline.py
pipeline_path = os.path.abspath("pipeline/pipeline.py")

# Cargar pipeline como módulo dinámico
spec = importlib.util.spec_from_file_location("pipeline_mod", pipeline_path)
pipeline_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline_mod)

# Ejecutar la función pipeline
if __name__ == "__main__":
    pipeline_mod.run_pipeline(
        image_path="resources/samples/000051652-1_2_1.png",
        model_path="models/chopin.pt",
        output_midi="output.mid"
    )
