import torch

from utils.data import get_mnist_loaders
from utils.train import train_model, evaluate_model, set_seed

from models.SoftmaxClassification import SoftmaxClassifier
from models.cnn_module import CNN
from models.ViT import ViT


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    
    train_loader, valid_loader, test_loader = get_mnist_loaders(batch_size=64)

    model_name = "vit"

    if model_name == "softmax":
        model = SoftmaxClassifier()

    elif model_name == "cnn":
        model = CNN()

    elif model_name == "vit":
        model = ViT()

    print("\nModel:")
    print(model)

    history = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader = valid_loader,
        epochs=10,
        learning_rate=0.001,
        optimizer_name="adam",
        device=device
    )

    test_loss, test_accuracy = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device
    )

    print("Final Result:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")


if __name__ == "__main__":
    main()