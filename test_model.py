
import sys
sys.path.insert(0, r".\original_code\potato-efficientViT\src")

import torch
from configurations import DEVICE, FEATURES
from models import EfficientNetV2B3ViT

print("Chargement du modèle...")

model = EfficientNetV2B3ViT(num_labels=FEATURES).to(DEVICE)
model.eval()

x = torch.randn(1, 3, 224, 224, device=DEVICE)

print("Test sur :", DEVICE)
print("Image entree :", tuple(x.shape))

with torch.no_grad():
    with torch.amp.autocast(
        device_type=DEVICE.type,
        enabled=(DEVICE.type == "cuda")
    ):
        output = model(x)

print("Sortie du modele :", tuple(output.shape))
print("Valeurs finies :", torch.isfinite(output).all().item())
print("Test GPU reussi !")

if DEVICE.type == "cuda":
    print(
        "Memoire GPU utilisee :",
        round(torch.cuda.memory_allocated() / 1024**3, 2),
        "Go"
    )
