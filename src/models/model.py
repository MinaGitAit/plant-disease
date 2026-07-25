
import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()


import torch.nn as nn
from torchvision import models

def build_model(num_classes):
    # Charger ResNet18 pré-entraîné sur ImageNet
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Geler toutes les couches existantes (on ne les ré-entraîne pas)
    for param in model.parameters():
        param.requires_grad = False

    # Remplacer la dernière couche (fc = fully connected)
    # model.fc.in_features = nombre de neurones entrant dans cette couche (512 pour ResNet18)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    # Cette nouvelle couche fc N'EST PAS gelée -> elle sera entraînée

    return model


if __name__ == "__main__":
    model = build_model(num_classes=38)
    print(model.fc)   # vérifie que la dernière couche a bien 38 sorties

    # Compter les paramètres entraînables vs totaux
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres totaux : {total_params:,}")
    print(f"Paramètres entraînables : {trainable_params:,}")