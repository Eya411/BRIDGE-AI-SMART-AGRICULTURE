
import torch
import torch.nn as nn
from timm import create_model
from transformers import ViTModel


class EfficientNetV2B3ViT(nn.Module):

    def __init__(self, num_labels=7):
        super().__init__()

        # ==========================================
        # 1. EfficientNetV2B3 : extraction features
        # ==========================================

        self.effnet = create_model(
            "tf_efficientnetv2_b3.in21k",
            pretrained=True,
            num_classes=0
        )

        # Geler initialement tous les paramètres
        for param in self.effnet.parameters():
            param.requires_grad = False

        # ==========================================
        # 2. Vision Transformer : extraction features
        # ==========================================

        self.vit = ViTModel.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )

        # Geler complètement ViT
        for param in self.vit.parameters():
            param.requires_grad = False

        # ==========================================
        # 3. Projection ViT : 768 -> 512
        # ==========================================

        self.vit_linear = nn.Linear(
            self.vit.config.hidden_size,
            512
        )

        # ==========================================
        # 4. Fusion des caractéristiques
        # ==========================================

        # EfficientNetV2B3 : 1536 features
        # ViT : 512 features

        self.dropout = nn.Dropout(0.2)

        self.classifier = nn.Linear(
            1536 + 512,
            num_labels
        )

        # ==========================================
        # 5. Initialisation des modes
        # ==========================================

        self.vit.eval()

    def train(self, mode=True):
        """
        EfficientNet peut passer en mode train.
        ViT reste toujours en mode eval.
        Les BatchNorm d'EfficientNet restent en eval
        pour préserver leurs statistiques pré-entraînées.
        """

        super().train(mode)

        # ViT complètement gelé
        self.vit.eval()

        # BatchNorm EfficientNet en eval
        for module in self.effnet.modules():
            if isinstance(
                module,
                nn.modules.batchnorm._BatchNorm
            ):
                module.eval()

        return self

    def forward(self, x):

        # ==========================================
        # 1. Extraction EfficientNet
        # ==========================================

        # PAS de torch.no_grad() ici :
        # nécessaire pour le fine-tuning des blocs débloqués.

        effnet_output = self.effnet.forward_features(x)

        # Global Average Pooling
        effnet_output = effnet_output.mean(dim=(2, 3))

        # ==========================================
        # 2. Extraction ViT
        # ==========================================

        # ViT reste complètement gelé
        with torch.no_grad():

            vit_output = self.vit(
                pixel_values=x
            ).last_hidden_state[:, 0]

        # Projection ViT : 768 -> 512
        # Cette couche reste entraînable

        vit_output = self.vit_linear(vit_output)

        # ==========================================
        # 3. Fusion des caractéristiques
        # ==========================================

        combined = torch.cat(
            (effnet_output, vit_output),
            dim=1
        )

        # ==========================================
        # 4. Classification
        # ==========================================

        combined = self.dropout(combined)

        output = self.classifier(combined)

        return output
