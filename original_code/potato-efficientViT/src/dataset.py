
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from configurations import (
    DATA,
    TEST_SIZE,
    VALI_SIZE,
    RANDOM_STATE,
    AUGMENT,
    BATCH_SIZE,
    DEVICE,
)
from utils import get_transforms_for_model


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)


class Dataset:

    def __init__(
        self,
        dataset=DATA,
        test_size=TEST_SIZE,
        vali_size=VALI_SIZE,
        random_state=RANDOM_STATE,
        augment=AUGMENT,
    ):
        self.dataset_name = dataset
        self.test_size = test_size
        self.vali_size = vali_size
        self.random_state = random_state
        self.augment = augment

        self.train_transforms = get_transforms_for_model(augment)
        self.other_transforms = get_transforms_for_model(False)

        # Emplacement du fichier de sauvegarde des splits
        project_root = Path(__file__).resolve().parents[3]
        self.split_file = project_root / "results" / "data_split.json"

    def prepare_dataset(self):

        # 1. Charger les images
        base_dataset = datasets.ImageFolder(self.dataset_name)

        targets = np.array(base_dataset.targets)
        indices = np.arange(len(targets))
        classes = base_dataset.classes

        if len(classes) != 7:
            raise ValueError(
                f"7 classes attendues, mais {len(classes)} trouvées : {classes}"
            )

        dataset_root = Path(self.dataset_name).resolve()

        # Inventaire des images et de leurs étiquettes
        all_samples = [
            {
                "path": str(
                    Path(path).resolve().relative_to(dataset_root)
                ),
                "label": int(label),
            }
            for path, label in base_dataset.samples
        ]

        logging.info(f"Classes détectées : {classes}")
        logging.info(f"Nombre total d'images : {len(base_dataset)}")

        self.split_file.parent.mkdir(parents=True, exist_ok=True)

        # 2. Charger un découpage déjà sauvegardé
        if self.split_file.exists():

            logging.info("Chargement du découpage existant...")

            with open(self.split_file, "r", encoding="utf-8") as file:
                saved_split = json.load(file)

            # Vérifier que le dataset n'a pas changé
            if saved_split["samples"] != all_samples:
                raise RuntimeError(
                    "Le dataset a changé depuis le dernier découpage. "
                    "Le fichier data_split.json ne correspond plus. "
                    "Vérifie les images avant de créer un nouveau split."
                )

            if saved_split["class_to_idx"] != base_dataset.class_to_idx:
                raise RuntimeError(
                    "Le mapping des classes a changé. "
                    "Vérifie le dataset."
                )

            train_indices = np.array(
                saved_split["train_indices"], dtype=int
            )
            vali_indices = np.array(
                saved_split["vali_indices"], dtype=int
            )
            test_indices = np.array(
                saved_split["test_indices"], dtype=int
            )

        # 3. Créer et sauvegarder le découpage initial
        else:

            logging.info("Création du découpage initial...")

            train_val_indices, test_indices = train_test_split(
                indices,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=targets,
            )

            train_indices, vali_indices = train_test_split(
                train_val_indices,
                test_size=self.vali_size,
                random_state=self.random_state,
                stratify=targets[train_val_indices],
            )

            saved_split = {
                "random_state": self.random_state,
                "test_size": self.test_size,
                "validation_size": self.vali_size,
                "classes": classes,
                "class_to_idx": base_dataset.class_to_idx,
                "samples": all_samples,
                "train_indices": train_indices.tolist(),
                "vali_indices": vali_indices.tolist(),
                "test_indices": test_indices.tolist(),
            }

            with open(
                self.split_file, "w", encoding="utf-8"
            ) as file:
                json.dump(
                    saved_split,
                    file,
                    indent=4,
                    ensure_ascii=False,
                )

            logging.info(
                f"Découpage sauvegardé : {self.split_file}"
            )

        # 4. Vérifier que chaque image appartient à un seul ensemble
        all_split_indices = np.concatenate([
            train_indices,
            vali_indices,
            test_indices,
        ])

        if (
            len(all_split_indices) != len(indices)
            or len(np.unique(all_split_indices)) != len(indices)
            or not np.array_equal(
                np.sort(all_split_indices), indices
            )
        ):
            raise ValueError(
                "Le découpage contient des doublons ou des images manquantes."
            )

        # 5. Créer trois datasets indépendants
        train_base = datasets.ImageFolder(
            self.dataset_name,
            transform=self.train_transforms,
        )

        vali_base = datasets.ImageFolder(
            self.dataset_name,
            transform=self.other_transforms,
        )

        test_base = datasets.ImageFolder(
            self.dataset_name,
            transform=self.other_transforms,
        )

        train_dataset = Subset(
            train_base, train_indices.tolist()
        )

        vali_dataset = Subset(
            vali_base, vali_indices.tolist()
        )

        test_dataset = Subset(
            test_base, test_indices.tolist()
        )

        # 6. Afficher les tailles
        logging.info(f"Images entraînement : {len(train_dataset)}")
        logging.info(f"Images validation : {len(vali_dataset)}")
        logging.info(f"Images test : {len(test_dataset)}")

        # 7. DataLoaders
        use_pin_memory = DEVICE.type == "cuda"

        train_dl = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=use_pin_memory,
        )

        vali_dl = DataLoader(
            vali_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=use_pin_memory,
        )

        test_dl = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=use_pin_memory,
        )

        logging.info("Chargement des datasets terminé.")
        logging.info(f"Fichier de découpage : {self.split_file}")

        return train_dl, vali_dl, test_dl
