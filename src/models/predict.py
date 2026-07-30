import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

import sys
import torch
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.model import build_model
from data.preprocess import get_dataloaders

IMG_SIZE = (224, 224)

def load_model(num_classes, device):
    model = build_model(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load("models/model.pth", map_location=device))
    model.eval()
    return model

def predict_image(image_path, model, classes, device):
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)  # ajoute une dimension "batch" de 1

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_class = classes[predicted_idx.item()]
    return predicted_class, confidence.item()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Récupérer la liste des classes (dans le même ordre que l'entraînement)
    _, _, classes = get_dataloaders()

    model = load_model(num_classes=len(classes), device=device)

    # Remplacez par le chemin d'une vraie image de test
    image_path = r"C:\Users\Hp\Desktop\plant-disease-mlops\data\test\TomatoYellowCurlVirus1.JPG"

    predicted_class, confidence = predict_image(image_path, model, classes, device)
    print(f"Prédiction : {predicted_class}")
    print(f"Confiance : {confidence*100:.2f}%")