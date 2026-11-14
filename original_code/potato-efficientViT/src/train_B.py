import sys
import json
import csv
import random

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from sklearn.metrics import accuracy_score

from configurations_B import *
from utils import get_transforms_for_model
from models_B import EfficientNetV2B3ViT_B


# =====================================================
# 1. REPRODUCTIBILITE
# =====================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =====================================================
# 2. CHARGEMENT DU DATASET ET DU SPLIT
# =====================================================

def create_dataloaders():

    print("Chargement du dataset...")

    train_transform = get_transforms_for_model(True)
    eval_transform = get_transforms_for_model(False)

    train_full = datasets.ImageFolder(
        DATA,
        transform=train_transform
    )

    eval_full = datasets.ImageFolder(
        DATA,
        transform=eval_transform
    )

    # Vérifier que les deux datasets ont le même ordre.
    if train_full.samples != eval_full.samples:
        raise RuntimeError(
            "L'ordre des images diffère entre les datasets."
        )

    if train_full.classes != CLASSES:
        raise RuntimeError(
            "Les classes du dataset ne correspondent pas à celles de B."
        )

    with open(SPLIT_FILE, "r", encoding="utf-8") as f:
        split = json.load(f)

    if split["classes"] != train_full.classes:
        raise RuntimeError(
            "Les classes de data_split.json ne correspondent pas."
        )

    train_indices = split["train_indices"]
    vali_indices = split["vali_indices"]
    test_indices = split["test_indices"]

    if (
        len(train_indices)
        + len(vali_indices)
        + len(test_indices)
        != len(train_full)
    ):
        raise RuntimeError(
            "Le nombre d'indices du split est incorrect."
        )

    train_dataset = Subset(
        train_full,
        train_indices
    )

    vali_dataset = Subset(
        eval_full,
        vali_indices
    )

    test_dataset = Subset(
        eval_full,
        test_indices
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    vali_loader = DataLoader(
        vali_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    print("\n========== DATASET B ==========")
    print("Train :", len(train_dataset))
    print("Validation :", len(vali_dataset))
    print("Test :", len(test_dataset))
    print("Classes :", train_full.classes)
    print("===============================\n")

    return train_loader, vali_loader, test_loader


# =====================================================
# 3. CREATION ET CHARGEMENT DU MODELE
# =====================================================

def create_model_B():

    print("Création du modèle B...")

    model = EfficientNetV2B3ViT_B(
        num_labels=NUM_CLASSES
    )

    print("Chargement du checkpoint A...")

    checkpoint = torch.load(
        BASELINE_CHECKPOINT,
        map_location="cpu",
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    print("Checkpoint A chargé avec succès.")

    model = model.to(DEVICE)

    print("\n========== PARAMETRES ENTRAINABLES ==========")

    total_trainable = 0

    for name, param in model.named_parameters():

        if param.requires_grad:

            print(name, param.numel())

            total_trainable += param.numel()

    print("TOTAL :", f"{total_trainable:,}")

    print("=============================================\n")

    return model


# =====================================================
# 4. OPTIMISEUR
# =====================================================

def create_optimizer_B(model):

    efficientnet_params = []
    head_params = []

    for name, param in model.named_parameters():

        if not param.requires_grad:
            continue

        if name.startswith("effnet."):
            efficientnet_params.append(param)

        elif (
            name.startswith("vit_linear.")
            or name.startswith("classifier.")
        ):
            head_params.append(param)

    if not efficientnet_params:
        raise RuntimeError(
            "Aucun paramètre EfficientNet n'est entraînable."
        )

    if not head_params:
        raise RuntimeError(
            "Aucun paramètre de la tête n'est entraînable."
        )

    optimizer = torch.optim.Adam(
        [
            {
                "params": efficientnet_params,
                "lr": LR_EFFICIENTNET
            },
            {
                "params": head_params,
                "lr": LR_HEAD
            }
        ],
        weight_decay=WEIGHT_DECAY
    )

    return optimizer


# =====================================================
# 5. ENTRAINEMENT D'UNE EPOCH
# =====================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler
):

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    optimizer.zero_grad(set_to_none=True)

    number_of_batches = len(loader)

    for batch_idx, (images, labels) in enumerate(loader):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        # Taille du groupe d'accumulation courant.
        group_number = batch_idx // GRAD_ACCUM_STEPS

        group_start = group_number * GRAD_ACCUM_STEPS

        group_size = min(
            GRAD_ACCUM_STEPS,
            number_of_batches - group_start
        )

        with torch.autocast(
            device_type=DEVICE.type,
            enabled=USE_AMP
        ):

            outputs = model(images)

            loss = criterion(outputs, labels)

            scaled_loss = loss / group_size

        scaler.scale(scaled_loss).backward()

        # Mise à jour après accumulation.
        should_update = (
            (batch_idx + 1) % GRAD_ACCUM_STEPS == 0
            or batch_idx + 1 == number_of_batches
        )

        if should_update:

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            scaler.step(optimizer)

            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

    epoch_loss = total_loss / total_samples

    epoch_accuracy = total_correct / total_samples

    return epoch_loss, epoch_accuracy


# =====================================================
# 6. VALIDATION
# =====================================================

@torch.no_grad()
def validate(model, loader, criterion):

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        with torch.autocast(
            device_type=DEVICE.type,
            enabled=USE_AMP
        ):

            outputs = model(images)

            loss = criterion(outputs, labels)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

    return (
        total_loss / total_samples,
        total_correct / total_samples
    )


# =====================================================
# 7. SAUVEGARDE
# =====================================================

def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    val_loss,
    val_accuracy
):

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "num_classes": NUM_CLASSES,
            "classes": CLASSES,
            "seed": SEED,
            "source_checkpoint": str(BASELINE_CHECKPOINT)
        },
        path
    )


# =====================================================
# 8. PROGRAMME PRINCIPAL
# =====================================================

def main():

    set_seed(SEED)

    print_configuration()

    train_loader, vali_loader, _ = create_dataloaders()

    model = create_model_B()

    criterion = nn.CrossEntropyLoss()

    optimizer = create_optimizer_B(model)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP
    )

    best_val_loss = float("inf")

    epochs_without_improvement = 0

    # Créer le fichier CSV.
    with open(CSV_LOG_B, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow([
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
            "lr_effnet",
            "lr_head"
        ])

    print("\n========== DEBUT FINE-TUNING B ==========\n")

    for epoch in range(1, EPOCHS + 1):

        print(f"\nEpoch {epoch}/{EPOCHS}")

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler
        )

        val_loss, val_accuracy = validate(
            model,
            vali_loader,
            criterion
        )

        scheduler.step(val_loss)

        lr_effnet = optimizer.param_groups[0]["lr"]

        lr_head = optimizer.param_groups[1]["lr"]

        print(f"Train loss      : {train_loss:.4f}")
        print(f"Train accuracy  : {train_accuracy:.4f}")
        print(f"Validation loss : {val_loss:.4f}")
        print(f"Validation acc  : {val_accuracy:.4f}")
        print(f"LR EfficientNet : {lr_effnet:.8f}")
        print(f"LR Head         : {lr_head:.8f}")

        with open(
            CSV_LOG_B,
            "a",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                epoch,
                train_loss,
                train_accuracy,
                val_loss,
                val_accuracy,
                lr_effnet,
                lr_head
            ])

        # Sauvegarder le dernier état.
        save_checkpoint(
            LAST_CHECKPOINT_B,
            epoch,
            model,
            optimizer,
            scheduler,
            val_loss,
            val_accuracy
        )

        # Sauvegarder le meilleur état.
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            epochs_without_improvement = 0

            save_checkpoint(
                BEST_CHECKPOINT_B,
                epoch,
                model,
                optimizer,
                scheduler,
                val_loss,
                val_accuracy
            )

            print("Nouveau meilleur checkpoint B sauvegardé.")

        else:

            epochs_without_improvement += 1

        # Early stopping.
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:

            print("\nEarly stopping déclenché.")

            break

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    print("\n========== ENTRAINEMENT B TERMINE ==========")

    print("Best checkpoint :", BEST_CHECKPOINT_B)

    print("Last checkpoint :", LAST_CHECKPOINT_B)

    print("CSV :", CSV_LOG_B)

    print("Meilleure validation loss :", best_val_loss)


if __name__ == "__main__":
    main()