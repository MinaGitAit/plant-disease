import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import mlflow

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.preprocess import get_dataloaders
from models.model import build_model


def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation de : {device}")

    # Charger les données (on utilise valid comme jeu d'évaluation ici)
    train_loader, valid_loader, classes = get_dataloaders()

    # Charger le modèle entraîné
    model = build_model(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load("models/model.pth", map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    print("Évaluation en cours...")
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # --- Rapport de classification (precision, recall, F1-score par classe) ---
    report = classification_report(all_labels, all_preds, target_names=classes, digits=4)
    print(report)

    # Sauvegarder le rapport dans un fichier texte
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    # --- Matrice de confusion ---
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=False, cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Classe prédite")
    plt.ylabel("Vraie classe")
    plt.title("Matrice de confusion")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("metrics/confusion_matrix.png", dpi=150)
    print("Matrice de confusion sauvegardée dans metrics/confusion_matrix.png")

    # --- Log dans MLflow ---
    mlflow.set_experiment("plant-disease-resnet18")
    with mlflow.start_run(run_name="evaluation_finale"):
        mlflow.log_artifact("metrics/classification_report.txt")
        mlflow.log_artifact("metrics/confusion_matrix.png")

        # Extraire l'accuracy globale et le F1-score moyen depuis le rapport
        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average="macro")
        f1_weighted = f1_score(all_labels, all_preds, average="weighted")

        mlflow.log_metric("final_accuracy", acc)
        mlflow.log_metric("final_f1_macro", f1_macro)
        mlflow.log_metric("final_f1_weighted", f1_weighted)

        print(f"\nAccuracy globale : {acc:.4f}")
        print(f"F1-score macro (moyenne simple par classe) : {f1_macro:.4f}")
        print(f"F1-score weighted (pondéré par nb d'images) : {f1_weighted:.4f}")


if __name__ == "__main__":
    evaluate_model()