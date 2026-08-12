import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.preprocess import get_dataloaders


def test_dataloaders_creation():
    """Vérifie que les DataLoaders se créent sans erreur et contiennent des données"""
    train_loader, valid_loader, classes = get_dataloaders()

    assert len(classes) == 38, "Le nombre de classes devrait être 38"
    assert len(train_loader.dataset) > 0, "Le dataset de train ne doit pas être vide"
    assert len(valid_loader.dataset) > 0, "Le dataset de valid ne doit pas être vide"


def test_batch_shape():
    """Vérifie que chaque batch a bien la forme attendue"""
    train_loader, _, _ = get_dataloaders()
    images, labels = next(iter(train_loader))

    assert images.shape[1:] == (3, 224, 224), "Chaque image doit être en 3x224x224"
    assert images.shape[0] == labels.shape[0], "Le nombre d'images doit correspondre au nombre de labels"