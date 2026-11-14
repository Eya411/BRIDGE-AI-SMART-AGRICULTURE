
from torchvision import transforms


def get_transforms_for_model(augment):
    """
    Retourne les transformations des images.
    augment=True  : transformations pour l'entraînement.
    augment=False : transformations pour la validation et le test.
    """

    if augment:
        data_transforms = transforms.Compose([
            transforms.Resize(256),

            transforms.RandomResizedCrop(
                224,
                scale=(0.95, 1.05),
                ratio=(0.75, 1.33),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),

            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),

            transforms.RandomRotation(40),

            transforms.ColorJitter(
                saturation=0.8,
                hue=0.021,
            ),

            transforms.RandomAffine(
                degrees=0,
                translate=(0.13, 0.13),
                scale=(0.8, 1.2),
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ])

    else:
        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),

            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ])

    return data_transforms
