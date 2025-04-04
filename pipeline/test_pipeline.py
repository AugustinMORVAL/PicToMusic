import sys
import os
import importlib.util

# Add root project path to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Dynamically import pipeline.py as a module
pipeline_path = os.path.join(project_root, "pipeline", "pipeline.py")
spec = importlib.util.spec_from_file_location("pipeline_module", pipeline_path)
pipeline_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline_mod)

# Define paths
image_path = "resources/samples/000051652-1_2_1.png"
model_notes = "models/chopin.pt"
model_symbols = "models/bach.pt"
output_midi = "output.mid"

# Run pipeline
pipeline_mod.run_pipeline(
    image_path=image_path,
    model_notes_path=model_notes,
    model_symbols_path=model_symbols,
    output_midi=output_midi
)
