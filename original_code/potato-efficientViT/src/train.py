
import csv
import logging
from pathlib import Path

import torch
import torch.nn as nn
import wandb

from configurations import (
    DEVICE,
    EPOCHS,
    BATCH_SIZE,
    GRAD_ACCUM_STEPS,
    FEATURES,
    USE_AMP,
    BEST_CHECKPOINT,
    LAST_CHECKPOINT,
    CSV_LOG,
    create_optimizer,
    create_scheduler,
    initialize_wandb,
)

from dataset import Dataset
from models import EfficientNetV2B3ViT


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class Trainer:

    def __init__(self, model, optimizer, scheduler, train_loader, vali_loader):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_loader = train_loader
        self.vali_loader = vali_loader

        self.criterion = nn.CrossEntropyLoss(reduction="sum")

        self.accum_steps = GRAD_ACCUM_STEPS
        self.amp_enabled = USE_AMP and DEVICE.type == "cuda"

        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.amp_enabled,
        )

        self.best_val_loss = float("inf")

        Path(BEST_CHECKPOINT).parent.mkdir(
            parents=True, exist_ok=True
        )

        Path(LAST_CHECKPOINT).parent.mkdir(
            parents=True, exist_ok=True
        )

        Path(CSV_LOG).parent.mkdir(
            parents=True, exist_ok=True
        )

    def train_one_epoch(self):

        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        num_batches = len(self.train_loader)

        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, (images, labels) in enumerate(self.train_loader):

            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            # Taille du groupe d'accumulation actuel
            group_start = (
                batch_idx // self.accum_steps
            ) * self.accum_steps

            group_batches = min(
                self.accum_steps,
                num_batches - group_start,
            )

            group_samples = min(
                group_batches * BATCH_SIZE,
                len(self.train_loader.dataset)
                - group_start * BATCH_SIZE,
            )

            # Forward pass avec mixed precision si disponible
            with torch.amp.autocast(
                device_type=DEVICE.type,
                enabled=self.amp_enabled,
            ):

                logits = self.model(images)

                loss_sum = self.criterion(logits, labels)

                # Normalisation correcte des gradients
                loss = loss_sum / group_samples

            self.scaler.scale(loss).backward()

            # Mise à jour après accumulation
            end_of_group = (
                (batch_idx + 1) % self.accum_steps == 0
                or batch_idx + 1 == num_batches
            )

            if end_of_group:

                self.scaler.step(self.optimizer)
                self.scaler.update()

                self.optimizer.zero_grad(set_to_none=True)

            # Calcul des métriques
            total_loss += loss_sum.item()

            predictions = logits.argmax(dim=1)

            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples += labels.size(0)

        average_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return average_loss, accuracy

    def validate(self):

        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():

            for images, labels in self.vali_loader:

                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                with torch.amp.autocast(
                    device_type=DEVICE.type,
                    enabled=self.amp_enabled,
                ):

                    logits = self.model(images)

                    loss_sum = self.criterion(logits, labels)

                total_loss += loss_sum.item()

                predictions = logits.argmax(dim=1)

                total_correct += (
                    predictions == labels
                ).sum().item()

                total_samples += labels.size(0)

        average_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return average_loss, accuracy

    def save_checkpoint(self, path, epoch, val_loss, val_acc):

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "num_classes": FEATURES,
        }

        torch.save(checkpoint, path)

    def train(self, num_epochs):

        history = []

        for epoch in range(num_epochs):

            logging.info(
                f"========== EPOCH {epoch + 1}/{num_epochs} =========="
            )

            # Entraînement
            train_loss, train_acc = self.train_one_epoch()

            # Validation
            val_loss, val_acc = self.validate()

            # Scheduler
            self.scheduler.step(val_loss)

            current_lr = self.optimizer.param_groups[0]["lr"]

            logging.info(
                f"Train Loss: {train_loss:.4f} | "
                f"Train Accuracy: {train_acc:.4f}"
            )

            logging.info(
                f"Validation Loss: {val_loss:.4f} | "
                f"Validation Accuracy: {val_acc:.4f}"
            )

            logging.info(f"Learning rate: {current_lr}")

            # Historique
            epoch_results = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "learning_rate": current_lr,
            }

            history.append(epoch_results)

            # Sauvegarder le dernier modèle
            self.save_checkpoint(
                LAST_CHECKPOINT,
                epoch + 1,
                val_loss,
                val_acc,
            )

            # Sauvegarder le meilleur modèle selon la loss validation
            if val_loss < self.best_val_loss:

                self.best_val_loss = val_loss

                self.save_checkpoint(
                    BEST_CHECKPOINT,
                    epoch + 1,
                    val_loss,
                    val_acc,
                )

                logging.info("Nouveau meilleur modèle sauvegardé.")

            # Enregistrement CSV
            with open(
                CSV_LOG,
                "w",
                newline="",
                encoding="utf-8-sig",
            ) as file:

                writer = csv.DictWriter(
                    file,
                    fieldnames=[
                        "epoch",
                        "train_loss",
                        "train_accuracy",
                        "val_loss",
                        "val_accuracy",
                        "learning_rate",
                    ],
                )

                writer.writeheader()
                writer.writerows(history)

            # Weights & Biases
            if wandb.run is not None:

                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "learning_rate": current_lr,
                })

        logging.info("Entraînement terminé.")
        logging.info(f"Meilleure validation loss : {self.best_val_loss:.4f}")



def main():

    logging.info("Initialisation du projet...")

    # 1. Initialisation de Weights & Biases
    initialize_wandb()

    # 2. Chargement des données
    dataset = Dataset()

    train_loader, vali_loader, test_loader = (
        dataset.prepare_dataset()
    )

    # 3. Création du modèle AVANT son utilisation
    model = EfficientNetV2B3ViT(
        num_labels=FEATURES
    ).to(DEVICE)

    logging.info("Modèle créé avec succès.")

    # 4. Vérification de la sortie du modèle
    model.eval()

    dummy_input = torch.randn(
        2, 3, 224, 224,
        device=DEVICE
    )

    with torch.no_grad():
        dummy_output = model(dummy_input)

    logging.info(
        f"Forme de sortie du modèle : {dummy_output.shape}"
    )

    if dummy_output.ndim != 2 or dummy_output.shape[1] != FEATURES:
        raise ValueError(
            f"Sortie incorrecte : {dummy_output.shape}. "
            f"Le modèle doit produire [batch, {FEATURES}]."
        )

    logging.info("Vérification du modèle réussie.")

    # 5. Passage en mode entraînement
    model.train()

    # 6. Création de l'optimiseur et du scheduler
    optimizer = create_optimizer(model)
    scheduler = create_scheduler(optimizer)

    # 7. Informations de configuration
    logging.info("Modèle : EfficientNetV2B3 + ViT")
    logging.info(f"Nombre de classes : {FEATURES}")
    logging.info(f"Device : {DEVICE}")
    logging.info(f"Epochs : {EPOCHS}")
    logging.info(f"Batch physique : {BATCH_SIZE}")
    logging.info(f"Accumulation : {GRAD_ACCUM_STEPS}")
    logging.info(f"AMP activé : {USE_AMP}")

    # 8. Création du Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        vali_loader=vali_loader,
    )

    # 9. Entraînement
    trainer.train(EPOCHS)

    # 10. Fermeture de WandB
    wandb.finish()


if __name__ == "__main__":
    main()
