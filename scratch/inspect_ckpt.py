import sys
import torch
sys.path.insert(0, ".")
from src.models.resnet50 import get_resnet50_model

ckpt_path = "results/models/resnet50/best_model.pth"
checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    ckpt_state = checkpoint["model_state_dict"]
else:
    ckpt_state = checkpoint

model = get_resnet50_model(num_classes=4)

print("Checkpoint keys sample:", list(ckpt_state.keys())[:5])
print("Model keys sample:     ", list(model.state_dict().keys())[:5])

try:
    model.load_state_dict(ckpt_state)
    print("Direct load: SUCCESS!")
except Exception as e:
    print("Direct load failed:", e)

try:
    model.backbone.load_state_dict(ckpt_state)
    print("Backbone load: SUCCESS!")
except Exception as e:
    print("Backbone load failed:", e)
