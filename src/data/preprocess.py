from torchvision import datasets, transforms
from torch.utils.data import DataLoader

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

valid_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

def get_dataloaders():
    train_dataset = datasets.ImageFolder("data/raw/train", transform=train_transform)
    valid_dataset = datasets.ImageFolder("data/raw/valid", transform=valid_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, valid_loader, train_dataset.classes


if __name__ == "__main__":
    train_loader, valid_loader, classes = get_dataloaders()

    print(f"Nombre de classes détectées : {len(classes)}")
    print(f"Images de train : {len(train_loader.dataset)}")
    print(f"Images de valid : {len(valid_loader.dataset)}")

    images, labels = next(iter(train_loader))
    print(f"Forme d'un batch d'images : {images.shape}")
    print(f"Forme d'un batch de labels : {labels.shape}")