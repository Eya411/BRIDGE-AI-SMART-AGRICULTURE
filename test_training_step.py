
import sys
sys.path.insert(0, r".\original_code\potato-efficientViT\src")

import torch
import torch.nn as nn

from configurations import (
    DEVICE,
    FEATURES,
    USE_AMP,
    create_optimizer,
)
from models import EfficientNetV2B3ViT

print("Construction du modèle...")

model = EfficientNetV2B3ViT(num_labels=FEATURES).to(DEVICE)
model.train()

optimizer = create_optimizer(model)
criterion = nn.CrossEntropyLoss()

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=USE_AMP
)

# Simulation d'un batch de 4 images
images = torch.randn(4, 3, 224, 224, device=DEVICE)
labels = torch.randint(0, FEATURES, (4,), device=DEVICE)

optimizer.zero_grad(set_to_none=True)

print("Début du pas d'entraînement...")

with torch.amp.autocast(
    device_type=DEVICE.type,
    enabled=USE_AMP
):
    outputs = model(images)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

print("\nPAS D'ENTRAÎNEMENT RÉUSSI")
print("Batch :", images.shape[0])
print("Sortie :", tuple(outputs.shape))
print("Loss :", loss.item())
print("Gradients calculés :", any(
    p.grad is not None
    for p in model.parameters()
    if p.requires_grad
))

if DEVICE.type == "cuda":
    print(
        "Mémoire GPU utilisée :",
        round(torch.cuda.memory_allocated() / 1024**3, 2),
        "Go"
    )
