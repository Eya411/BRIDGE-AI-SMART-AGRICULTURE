
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from configurations_B import (
    DATA,
    SPLIT_FILE,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    DEVICE,
    CLASSES,
    NUM_CLASSES
)

from utils import get_transforms_for_model
from models_B import EfficientNetV2B3ViT_B


# =====================================================
# 1. CHARGEMENT DU JEU DE TEST
# =====================================================

def create_test_loader():

    transform = get_transforms_for_model(False)

    dataset = datasets.ImageFolder(
        DATA,
        transform=transform
    )

    if dataset.classes != CLASSES:
        raise RuntimeError(
            "Les classes du dataset ne correspondent pas."
        )

    if len(dataset.classes) != NUM_CLASSES:
        raise RuntimeError(
            "Nombre de classes incorrect."
        )

    with open(SPLIT_FILE, "r", encoding="utf-8") as f:
        split = json.load(f)

    if split["classes"] != dataset.classes:
        raise RuntimeError(
            "Le mapping des classes est différent."
        )

    test_indices = split["test_indices"]

    if len(test_indices) != 308:
        raise RuntimeError(
            f"Le test devrait contenir 308 images, "
            f"mais contient {len(test_indices)}."
        )

    test_dataset = Subset(
        dataset,
        test_indices
    )

    loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    print("\n========== TEST DATASET ==========")
    print("Nombre d'images :", len(test_dataset))
    print("Classes :", dataset.classes)
    print("Mapping :", dataset.class_to_idx)
    print("==================================\n")

    return loader


# =====================================================
# 2. CREATION DU MODELE
# =====================================================

def load_evaluation_model(checkpoint_path):

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    model = EfficientNetV2B3ViT_B(
        num_labels=NUM_CLASSES
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    if "model_state_dict" not in checkpoint:
        raise RuntimeError(
            "Checkpoint invalide : model_state_dict absent."
        )

    # Chargement strict des poids.
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    if "classes" in checkpoint:
        if checkpoint["classes"] != CLASSES:
            raise RuntimeError(
                "Les classes du checkpoint ne correspondent pas."
            )

    model = model.to(DEVICE)
    model.eval()

    print("Checkpoint chargé :", checkpoint_path)
    print("Epoch checkpoint :", checkpoint.get("epoch", "Inconnue"))
    print("Validation loss :", checkpoint.get("val_loss", "Inconnue"))
    print("Validation accuracy :", checkpoint.get(
        "val_accuracy", "Inconnue"
    ))

    return model


# =====================================================
# 3. PREDICTIONS
# =====================================================

@torch.no_grad()
def predict_model(model, loader):

    all_labels = []
    all_predictions = []
    all_probabilities = []

    model.eval()

    for images, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        with torch.autocast(
            device_type=DEVICE.type,
            enabled=(DEVICE.type == "cuda")
        ):

            outputs = model(images)

        if isinstance(outputs, (tuple, list)):
            outputs = outputs[0]

        probabilities = torch.softmax(
            outputs.float(),
            dim=1
        )

        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.numpy().tolist())

        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy().tolist()
        )

    return (
        all_labels,
        all_predictions,
        all_probabilities
    )
