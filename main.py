import torch

from utils.data import get_mnist_loaders
from utils.train import train_model, evaluate_model

from models.SoftmaxClassification import SoftmaxClassifier
from models.cnn_module import CNN
from models.ViT import ViT


def main():
    # =========================
    # 1. chọn phần cứng
    # =========================
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # =========================
    # 2. tải MNIST
    # =========================

    train_loader, test_loader = get_mnist_loaders(
        batch_size=64
    )

    # =========================
    # 3. Chọn model
    # =========================

    model_name = "cnn"

    if model_name == "softmax":
        model = SoftmaxClassifier()

    elif model_name == "cnn":
        model = CNN()

    elif model_name == "vit":
        model = ViT()

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )


    print("\nModel:")
    print(model)


    # =========================
    # 4. Train
    # =========================

    history = train_model(
        model=model,
        train_loader=train_loader,
        epochs=10,
        learning_rate=0.001,
        optimizer_name="adam",
        device=device
    )


    # =========================
    # 5. Test
    # =========================

    test_loss, test_accuracy = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device
    )


    # =========================
    # 6. Kết quả
    # =========================

    print("\n========================")
    print("Final Result")
    print("========================")

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")


if __name__ == "__main__":
    main()