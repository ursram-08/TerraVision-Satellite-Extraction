import torch
from pathlib import Path
from satseg.models import build_model

files = {
    ("roads", "unet"): "best_road_unet.pth",
    ("roads", "attention_unet"): "best_road_attention_unet.pth",
    ("buildings", "unet"): "best_unet.pth",
    ("buildings", "attention_unet"): "best_attention_unet.pth",
}

for (task, name), filename in files.items():
    print(f"\nTesting {filename}...")

    model = build_model(name, task)

    checkpoint = torch.load(
        Path("checkpoints") / filename,
        map_location="cpu",
        weights_only=True
    )

    if "model_state" in checkpoint:
        checkpoint = checkpoint["model_state"]

    model.load_state_dict(checkpoint, strict=True)

    print(f"SUCCESS: {filename}")

print("\nALL FOUR CHECKPOINTS LOADED SUCCESSFULLY!")