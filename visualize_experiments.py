import os
import time
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from models.SoftmaxClassification import SoftmaxClassifier
from models.cnn_module import CNN
from utils.data import get_mnist_loaders
from utils.train import train_model, evaluate_model


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESULT_DIR = "results"

# Your already trained ViT result.
# We do NOT train ViT again because you said it is too slow.
VIT_TEST_ACCURACY = 97.90
VIT_CHECKPOINT = "checkpoints/vit.pth"

# Recorded ViT training history from your previous run.
# ViT is not trained again in this visualization script.
VIT_HISTORY = {
    "train_loss": [0.5191, 0.1982, 0.1531, 0.1317, 0.1161, 0.1076, 0.1005, 0.0924, 0.0872, 0.0837],
    "train_accuracy": [83.15, 93.79, 95.10, 95.85, 96.42, 96.56, 96.83, 97.07, 97.30, 97.29],
    "valid_loss": [0.2020, 0.1439, 0.1235, 0.0999, 0.0998, 0.0888, 0.0842, 0.0842, 0.0837, 0.0764],
    "valid_accuracy": [93.93, 95.57, 96.16, 96.89, 96.91, 97.26, 97.40, 97.38, 97.48, 97.73]
}

# Number of images used in the visual comparison.
NUM_COMPARE_IMAGES = 10


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ============================================================
# BASIC UTILITIES
# ============================================================

def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def get_predictions(model, loader):
    model.to(DEVICE)
    model.eval()

    all_images = []
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(DEVICE))
            predictions = torch.argmax(outputs, dim=1)

            all_images.append(images.cpu())
            all_labels.append(labels.cpu())
            all_predictions.append(predictions.cpu())

    return (
        torch.cat(all_images),
        torch.cat(all_labels),
        torch.cat(all_predictions)
    )


def confusion_matrix(labels, predictions, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=int)

    for true_label, predicted_label in zip(labels, predictions):
        cm[int(true_label), int(predicted_label)] += 1

    return cm


# ============================================================
# 1. TRAIN SOFTMAX + CNN
# ============================================================

def train_two_models(train_loader, valid_loader, test_loader):

    models = {
        "Softmax": SoftmaxClassifier(),
        "CNN": CNN()
    }

    histories = {}
    results = {}
    predictions = {}

    for name, model in models.items():

        print("\n" + "=" * 60)
        print(f"Training {name}")
        print("=" * 60)
        print(model)

        start_time = time.perf_counter()

        history = train_model(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            optimizer_name="adam",
            device=DEVICE
        )

        training_time = time.perf_counter() - start_time

        test_loss, test_accuracy = evaluate_model(
            model=model,
            test_loader=test_loader,
            device=DEVICE
        )

        histories[name] = history

        results[name] = {
            "test_accuracy": test_accuracy,
            "test_loss": test_loss,
            "parameters": count_parameters(model),
            "training_time": training_time
        }

        _, labels, model_predictions = get_predictions(
            model,
            test_loader
        )

        predictions[name] = {
            "labels": labels,
            "predictions": model_predictions
        }

        # Save weights so that you can reuse them later.
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(
            model.state_dict(),
            f"checkpoints/{name.lower()}.pth"
        )

        print(f"{name} Test Accuracy: {test_accuracy:.2f}%")

    return models, histories, results, predictions


# ============================================================
# 2. LEARNING CURVES
# ============================================================

def plot_learning_curves(histories):

    epochs = range(1, EPOCHS + 1)

    # Loss
    plt.figure(figsize=(9, 6))

    for name, history in histories.items():
        plt.plot(
            epochs,
            history["train_loss"],
            marker="o",
            label=f"{name} Train"
        )

        plt.plot(
            epochs,
            history["valid_loss"],
            marker="o",
            linestyle="--",
            label=f"{name} Validation"
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.xticks(list(epochs))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULT_DIR, "train_vs_valid_loss.png"),
        dpi=300
    )
    plt.show()
    plt.close()

    # Accuracy
    plt.figure(figsize=(9, 6))

    for name, history in histories.items():
        plt.plot(
            epochs,
            history["train_accuracy"],
            marker="o",
            label=f"{name} Train"
        )

        plt.plot(
            epochs,
            history["valid_accuracy"],
            marker="o",
            linestyle="--",
            label=f"{name} Validation"
        )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training vs Validation Accuracy")
    plt.xticks(list(epochs))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULT_DIR, "train_vs_valid_accuracy.png"),
        dpi=300
    )
    plt.show()
    plt.close()


# ============================================================
# 3. TEST ACCURACY TABLE
# ============================================================

def print_test_accuracy_table(results):

    print("\n" + "=" * 70)
    print("TEST ACCURACY COMPARISON")
    print("=" * 70)

    print(
        f"{'Model':<15}"
        f"{'Test Loss':>15}"
        f"{'Test Accuracy':>20}"
        f"{'Parameters':>20}"
    )

    print("-" * 70)

    for name, result in results.items():
        print(
            f"{name:<15}"
            f"{result['test_loss']:>15.4f}"
            f"{result['test_accuracy']:>19.2f}%"
            f"{result['parameters']:>20,}"
        )

    print("=" * 70)


# ============================================================
# 4. CONFUSION MATRICES
# ============================================================

def plot_confusion_matrix(cm, model_name):

    plt.figure(figsize=(7, 6))

    plt.imshow(cm)
    plt.colorbar()

    plt.xticks(range(10))
    plt.yticks(range(10))

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {model_name}")

    for i in range(10):
        for j in range(10):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                fontsize=8
            )

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            RESULT_DIR,
            f"confusion_{model_name.lower()}.png"
        ),
        dpi=300
    )
    plt.show()
    plt.close()


def make_confusion_matrices(predictions):

    for name in ["Softmax", "CNN"]:
        labels = predictions[name]["labels"].numpy()
        preds = predictions[name]["predictions"].numpy()

        cm = confusion_matrix(labels, preds)
        plot_confusion_matrix(cm, name)


# ============================================================
# 5. VISUAL COMPARISON: SAME IMAGES, THREE MODELS
# ============================================================

def show_model_predictions(models, test_dataset):

    # Select the first NUM_COMPARE_IMAGES images from the test set.
    images = []
    labels = []

    for i in range(NUM_COMPARE_IMAGES):
        image, label = test_dataset[i]
        images.append(image)
        labels.append(label)

    images = torch.stack(images)
    labels = torch.tensor(labels)

    model_predictions = {}

    for name, model in models.items():
        model.eval()

        with torch.no_grad():
            outputs = model(images.to(DEVICE))
            model_predictions[name] = torch.argmax(
                outputs,
                dim=1
            ).cpu()

    # --------------------------------------------------------
    # One figure: same image, predictions from each model
    # --------------------------------------------------------

    model_names = list(models.keys())

    rows = 1 + len(model_names)

    plt.figure(
        figsize=(16, 3 * rows)
    )

    for col in range(NUM_COMPARE_IMAGES):

        # Image
        plt.subplot(rows, NUM_COMPARE_IMAGES, col + 1)
        plt.imshow(images[col].squeeze(), cmap="gray")
        plt.title(f"True: {labels[col].item()}")
        plt.axis("off")

    for row, model_name in enumerate(model_names, start=1):

        preds = model_predictions[model_name]

        for col in range(NUM_COMPARE_IMAGES):

            plt.subplot(
                rows,
                NUM_COMPARE_IMAGES,
                row * NUM_COMPARE_IMAGES + col + 1
            )

            plt.imshow(
                images[col].squeeze(),
                cmap="gray"
            )

            true_label = labels[col].item()
            predicted_label = preds[col].item()

            title = (
                f"{model_name}\n"
                f"Pred: {predicted_label}"
            )

            plt.title(title)
            plt.axis("off")

    plt.suptitle(
        "Prediction Comparison on the Same Test Images",
        fontsize=16
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULT_DIR,
            "prediction_comparison.png"
        ),
        dpi=300
    )

    plt.show()
    plt.close()


# ============================================================
# 6. MAIN
# ============================================================

def main():

    os.makedirs(RESULT_DIR, exist_ok=True)

    set_seed(SEED)

    print("Device:", DEVICE)
    print("Loading MNIST...")

    train_loader, valid_loader, test_loader = get_mnist_loaders(
        batch_size=BATCH_SIZE
    )

    # Train only Softmax and CNN.
    # ViT is not retrained because it is slow on CPU.
    models, histories, results, predictions = train_two_models(
        train_loader,
        valid_loader,
        test_loader
    )

    # Add the recorded ViT result and history.
    # We do not pretend to have ViT predictions without its weights.
    histories["ViT"] = VIT_HISTORY

    results["ViT"] = {
        "test_accuracy": VIT_TEST_ACCURACY,
        "test_loss": float("nan"),
        "parameters": None,
        "training_time": None
    }

    # --------------------------------------------------------
    # Experiment 1 + 2: Learning curves
    # --------------------------------------------------------

    plot_learning_curves(histories)

    # --------------------------------------------------------
    # Test accuracy table
    # --------------------------------------------------------

    print_test_accuracy_table(results)

    # --------------------------------------------------------
    # Experiment 3: Confusion matrices
    # --------------------------------------------------------

    make_confusion_matrices(predictions)

    # --------------------------------------------------------
    # Visual prediction comparison
    # --------------------------------------------------------

    show_model_predictions(
        models,
        test_loader.dataset
    )

    print("\nResults saved to:")
    print(os.path.abspath(RESULT_DIR))

    print("\nNote:")
    print(
        "ViT is not retrained in this script. "
        "The table uses your recorded ViT test accuracy = 97.90%."
    )
    print(
        "To include ViT predictions in the image comparison, "
        f"save its state_dict to: {VIT_CHECKPOINT}"
    )


if __name__ == "__main__":
    main()
