import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def train_model(model, train_dataset, val_dataset, device, epochs=5, batch_size=32, lr=1e-3):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    model.to(DEVICE)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            images, labels = batch[0], batch[1]
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        val_acc = evaluate(model, val_loader, DEVICE)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")


def evaluate(model, data_loader, device=DEVICE):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for batch in data_loader:
            images, labels = batch[0], batch[1]
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total