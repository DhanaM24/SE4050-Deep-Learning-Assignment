import json
import os
import sys
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for automated testing

import torch

notebook_path = "notebooks/resnet50_training.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells in notebook: {len(nb['cells'])}", flush=True)

gl = {}

for idx, cell in enumerate(nb["cells"], 1):
    cell_type = cell.get("cell_type")
    source = "".join(cell.get("source", []))
    print(f"\n--- Executing Cell {idx} ({cell_type}) ---", flush=True)
    
    if cell_type == "code":
        try:
            exec(source, gl, gl)
        except Exception as e:
            print(f"ERROR executing Cell {idx}:\n{e}", flush=True)
            sys.exit(1)

print("\nNotebook execution completed top-to-bottom with ZERO errors!", flush=True)
