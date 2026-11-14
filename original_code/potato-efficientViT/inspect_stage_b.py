import json
from pathlib import Path
import torch
from torchvision import datasets
ROOT = Path(r"C:\Users\ASUS\Desktop\TERMINALE MASTERE EYA\iot\Projet\BRIDGE_AI_POTATO")
DATA = ROOT / "data"
BEST = ROOT / "results" / "checkpoints" / "Hybrid_EffNetViT_30epochs_seed42_best.pth"
LAST = ROOT / "results" / "checkpoints" / "Hybrid_EffNetViT_30epochs_seed42_last.pth"
print("\n========== 1. DATASET ==========")
dataset = datasets.ImageFolder(DATA)
print("Nombre total :", len(dataset))
print("Classes :", dataset.classes)
print("Mapping :", dataset.class_to_idx)
for class_name, index in dataset.class_to_idx.items():
    count = sum(1 for _, label in dataset.samples if label == index)
    print(f"{class_name}: {count} images")
print("\n========== 2. DATA SPLIT ==========")
split_file = ROOT / "results" / "data_split.json"
if split_file.exists():
    with open(split_file, encoding="utf-8") as f:
        split = json.load(f)
    print("Fichier :", split_file)
    print("Random state :", split.get("random_state"))
    print("Train :", len(split["train_indices"]))
    print("Validation :", len(split["vali_indices"]))
    print("Test :", len(split["test_indices"]))
    print("Classes du split :", split.get("classes"))
else:
    print("ATTENTION : data_split.json absent")
print("\n========== 3. CHECKPOINTS ==========")
for path in [BEST, LAST]:
    print(f"\n--- {path.name} ---")
    print("Existe :", path.exists())
    if not path.exists():
        continue
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )
    print("Type :", type(checkpoint).__name__)
    if isinstance(checkpoint, dict):
        print("Clés principales :", list(checkpoint.keys()))
        if "epoch" in checkpoint:
            print("Epoch :", checkpoint["epoch"])
        if "val_loss" in checkpoint:
            print("Validation loss :", checkpoint["val_loss"])
        if "val_accuracy" in checkpoint:
            print("Validation accuracy :", checkpoint["val_accuracy"])
        state = checkpoint.get(
            "model_state_dict",
            checkpoint.get("state_dict", checkpoint)
        )
        if isinstance(state, dict):
            print("Nombre de paramètres :", len(state))
            print("\nPremières clés :")
            for key in list(state.keys())[:15]:
                value = state[key]
                if torch.is_tensor(value):
                    print(key, tuple(value.shape))
                else:
                    print(key, type(value).__name__)
            print("\nCouches de classification :")
            for key, value in state.items():
                if any(name in key for name in [
                    "classifier", "vit_linear"
                ]) and torch.is_tensor(value):
                    print(key, tuple(value.shape))
    del checkpoint
print("\n========== INSPECTION TERMINEE ==========")
