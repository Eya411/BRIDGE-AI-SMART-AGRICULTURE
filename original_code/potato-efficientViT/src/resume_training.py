
import torch
from configurations import MODELS, DEVICE, CRITERION
from dataset import Dataset

CHECKPOINT = "CHEMIN_VERS_TON_CHECKPOINT_LAST.pth"

TARGET_EPOCHS = 70
BATCH_ACCUMULATION = 16

def load_weights(model, path):
    checkpoint = torch.load(
        path,
        map_location=DEVICE,
        weights_only=False
    )

    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint

    model.load_state_dict(state, strict=True)

    return checkpoint

def main():
    model_name = "EffNetViT"
    model = MODELS[model_name].to(DEVICE)

    checkpoint = load_weights(model, CHECKPOINT)

    print("Checkpoint chargé correctement.")

    dataset = Dataset()
    train_loader, val_loader, test_loader = (
        dataset.prepare_dataset()
    )

    # IMPORTANT :
    # Reprendre ici le même optimizer et scheduler
    # que dans ton train.py original.
    #
    # Si les états de l'optimizer et du scheduler
    # existent dans le checkpoint, les restaurer.
    #
    # Sinon, les recréer avec les paramètres du baseline.

    print("Train batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    print("Prêt pour la reprise jusqu'à 70 époques.")

if __name__ == "__main__":
    main()