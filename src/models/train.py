import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

import yaml
import torch
import torch.nn as nn
import mlflow
import sys
import random
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.preprocess import get_dataloaders
from models.model import build_model


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)["train"]


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()  # mode entraînement (active dropout etc. si présent)
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()               # remet les gradients à zéro
        outputs = model(images)              # 1. forward pass
        loss = criterion(outputs, labels)    # 2. loss

        loss.backward()                      # 3. backward pass (calcule les gradients)
        optimizer.step()                     # 4. optimizer step (ajuste les poids)

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()  # mode évaluation (désactive dropout etc.)
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # pas besoin de calculer les gradients ici
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    params = load_params()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation de : {device}")

    train_loader, valid_loader, classes = get_dataloaders()

    # --- RÉDUCTION TEMPORAIRE POUR TEST RAPIDE ---
    SUBSET_SIZE = 500
    train_indices = random.sample(range(len(train_loader.dataset)), SUBSET_SIZE)
    valid_indices = random.sample(range(len(valid_loader.dataset)), 200)

    train_subset = Subset(train_loader.dataset, train_indices)
    valid_subset = Subset(valid_loader.dataset, valid_indices)

    train_loader = DataLoader(train_subset, batch_size=params["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_subset, batch_size=params["batch_size"], shuffle=False)
    print(f"Mode test rapide : {SUBSET_SIZE} images train, 200 images valid")
    # --- FIN RÉDUCTION TEMPORAIRE ---

    model = build_model(num_classes=len(classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    # On optimise SEULEMENT les paramètres non gelés (model.fc)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=params["learning_rate"])

    mlflow.set_experiment("plant-disease-resnet18")

    with mlflow.start_run():
        mlflow.log_param("epochs", params["epochs"])
        mlflow.log_param("learning_rate", params["learning_rate"])
        mlflow.log_param("batch_size", params["batch_size"])
        mlflow.log_param("model", "resnet18_transfer_learning")
        mlflow.log_param("num_classes", len(classes))

        for epoch in range(params["epochs"]):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            valid_loss, valid_acc = evaluate(model, valid_loader, criterion, device)

            print(f"Epoch {epoch+1}/{params['epochs']} - "
                  f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f} - "
                  f"valid_loss: {valid_loss:.4f}, valid_acc: {valid_acc:.4f}")

            # Log des métriques à CHAQUE epoch (permet de voir la courbe dans MLflow)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("valid_loss", valid_loss, step=epoch)
            mlflow.log_metric("valid_accuracy", valid_acc, step=epoch)

        # Sauvegarder le modèle final
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/model.pth")
        mlflow.log_artifact("models/model.pth")

        print("Entraînement terminé. Modèle sauvegardé dans models/model.pth")


if __name__ == "__main__":
    main()