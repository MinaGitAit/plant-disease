import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from models.model import build_model


def test_model_output_shape():
    """Vérifie que le modèle produit une sortie de la bonne taille"""
    num_classes = 38
    model = build_model(num_classes=num_classes)
    model.eval()

    # Créer une fausse image (batch de 1, 3 canaux, 224x224)
    dummy_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (1, num_classes), f"La sortie doit être de forme (1, {num_classes})"


def test_frozen_layers():
    """Vérifie que les couches convolutives sont bien gelées"""
    model = build_model(num_classes=38)

    # Vérifier que la couche fc (dernière couche) est entraînable
    assert model.fc.weight.requires_grad == True, "La derniere couche doit etre entrainable"

    # Vérifier qu'une couche antérieure est bien gelée
    first_conv_param = next(model.conv1.parameters())
    assert first_conv_param.requires_grad == False, "Les couches convolutives doivent etre gelees"