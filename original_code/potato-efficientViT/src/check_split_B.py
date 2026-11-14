
import json
from collections import Counter
from pathlib import Path
from torchvision import datasets

from configurations_B import DATA, SPLIT_FILE

dataset = datasets.ImageFolder(DATA)

with open(SPLIT_FILE, "r", encoding="utf-8") as f:
    split = json.load(f)

targets = dataset.targets

for name, key in [
    ("TRAIN", "train_indices"),
    ("VALIDATION", "vali_indices"),
    ("TEST", "test_indices")
]:
    indices = split[key]
    counts = Counter(targets[i] for i in indices)

    print(f"\n===== {name} =====")
    print("Total :", len(indices))

    for class_id, class_name in enumerate(dataset.classes):
        count = counts[class_id]
        print(
            f"{class_name:15s}: {count:4d} "
            f"({100 * count / len(indices):.2f}%)"
        )
