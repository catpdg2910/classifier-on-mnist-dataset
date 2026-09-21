# utils/train.py
import torch
import torch.nn as nn
import torch.optim as optim


def train_model(model, train_loader, epochs=10, learning_rate=0.001, optimizer_name="adam", device="cpu"):
    model.to(device)
    loss_func = nn.CrossEntropyLoss()
    if optimizer_name.lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate
        )

    elif optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate*10
        )

    history = {
        "train_loss": [],
        "train_accuracy": []
    }

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)

            # Forward
            outputs = model(images)
            # Loss
            loss = loss_func(outputs, labels)
            # Backward
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            prediction = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (prediction == labels).sum().item()
            
        epoch_loss = total_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total

        history["train_loss"].append(epoch_loss)
        history["train_accuracy"].append(epoch_accuracy)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {epoch_loss:.4f} "
            f"Accuracy: {epoch_accuracy:.2f}%"
        )
    return history


def evaluate_model(model, test_loader, device="cpu"):
    model.to(device)
    model.eval()
    loss_func = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = loss_func(outputs, labels)
            total_loss += loss.item()
            prediction = torch.argmax(outputs, dim=1)
            
            total += labels.size(0)
            correct += (prediction == labels).sum().item()

    test_loss = total_loss / len(test_loader)
    test_accuracy = 100 * correct / total
    return test_loss, test_accuracy