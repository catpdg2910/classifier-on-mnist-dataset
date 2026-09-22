import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_model(
    model,
    train_loader,
    valid_loader,
    epochs=10,
    learning_rate=0.001,
    optimizer_name="adam",
    device="cpu"
):
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    if optimizer_name.lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate
        )
    elif optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate
        )
    else:
        raise ValueError(
            f"Unsupported optimizer: {optimizer_name}"
        )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "valid_loss": [],
        "valid_accuracy": []
    }

    for epoch in range(epochs):

        # ==================================
        # Training
        # ==================================

        model.train()

        total_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Forward
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            total_train_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss = total_train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total

        # ==================================
        # Validation
        # ==================================

        model.eval()

        total_valid_loss = 0.0
        valid_correct = 0
        valid_total = 0

        with torch.no_grad():

            for images, labels in valid_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)

                total_valid_loss += loss.item()

                _, predicted = torch.max(outputs, 1)

                valid_total += labels.size(0)
                valid_correct += (
                    predicted == labels
                ).sum().item()

        valid_loss = total_valid_loss / len(valid_loader)
        valid_accuracy = 100 * valid_correct / valid_total

        # ==================================
        # Save history
        # ==================================

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)

        history["valid_loss"].append(valid_loss)
        history["valid_accuracy"].append(valid_accuracy)

        # ==================================
        # Print result
        # ==================================

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_accuracy:.2f}% "
            f"Valid Loss: {valid_loss:.4f} "
            f"Valid Acc: {valid_accuracy:.2f}%"
        )

    return history


def evaluate_model(
    model,
    test_loader,
    device="cpu"):
    
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = total_loss / len(test_loader)
    test_accuracy = 100 * correct / total

    return test_loss, test_accuracy