import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

import sys
import io
import torch
from PIL import Image
from torchvision import transforms
from fastapi import FastAPI, UploadFile, File, HTTPException

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.model import build_model

IMG_SIZE = (224, 224)

# Liste des classes, dans le MÊME ORDRE que celui utilisé pendant l'entraînement
# (ImageFolder trie les dossiers par ordre alphabétique, donc on peut la reconstruire ainsi)
CLASSES = sorted(os.listdir("data/raw/train"))

app = FastAPI(title="Plant Disease Prediction API")

# Variables globales pour le modèle (chargé une seule fois au démarrage)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None


@app.on_event("startup")
def load_model():
    global model
    model = build_model(num_classes=len(CLASSES)).to(device)
    model.load_state_dict(torch.load("models/model.pth", map_location=device))
    model.eval()
    print(f"Modèle chargé avec succès sur {device}")


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Vérifier que c'est bien une image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image")

    # Lire l'image envoyée
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Appliquer les mêmes transformations que pendant l'entraînement/évaluation
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Prédiction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_class = CLASSES[predicted_idx.item()]

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence.item() * 100, 2)
    }