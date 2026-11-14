import torch
import torch.nn as nn

from timm import create_model
from transformers import ViTModel


class EfficientNetV2B3ViT_B(nn.Module):

    def __init__(self, num_labels=7):

        super().__init__()

        # 1. EfficientNetV2B3
        # Les poids seront chargés depuis le checkpoint A.
        self.effnet = create_model(
            "tf_efficientnetv2_b3.in21k",
            pretrained=False,
            num_classes=0
        )

        # 2. ViT
        # On conserve la même architecture que dans models.py.
        self.vit = ViTModel.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )

        # 3. Projection ViT
        self.vit_linear = nn.Linear(
            self.vit.config.hidden_size,
            512
        )

        # 4. Classification
        self.dropout = nn.Dropout(0.2)

        self.classifier = nn.Linear(
            1536 + 512,
            num_labels
        )

        # 5. Geler initialement tous les paramètres.
        for param in self.parameters():
            param.requires_grad = False

        # 6. Débloquer le dernier sous-bloc EfficientNet.
        for param in self.effnet.blocks[5][-1].parameters():
            param.requires_grad = True

        # 7. Garder BatchNorm gelées.
        for module in self.effnet.blocks[5][-1].modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                for param in module.parameters():
                    param.requires_grad = False

        # 8. Débloquer la projection et le classifieur.
        for param in self.vit_linear.parameters():
            param.requires_grad = True

        for param in self.classifier.parameters():
            param.requires_grad = True

        self.vit.eval()

    def train(self, mode=True):

        super().train(mode)

        # ViT toujours en mode évaluation.
        self.vit.eval()

        # BatchNorm EfficientNet toujours en mode évaluation.
        for module in self.effnet.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

        return self

    def forward(self, x):

        # Branche EfficientNet
        effnet_output = self.effnet.forward_features(x)
        effnet_output = effnet_output.mean(dim=(2, 3))

        # Branche ViT gelée
        with torch.no_grad():
            vit_output = self.vit(
                pixel_values=x
            ).last_hidden_state[:, 0]

        # Projection ViT
        vit_output = self.vit_linear(vit_output)

        # Fusion
        combined = torch.cat(
            (effnet_output, vit_output),
            dim=1
        )

        combined = self.dropout(combined)

        # Classification finale
        output = self.classifier(combined)

        return output