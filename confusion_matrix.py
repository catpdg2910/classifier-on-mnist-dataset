import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.SoftmaxClassification import SoftmaxClassifier
from models.cnn_module import CNN


# =========================
# 1. Configuration
# =========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = "./MNIST DATASET"

SOFTMAX_PATH = "./checkpoints/softmax.pth"
CNN_PATH = "./checkpoints/cnn.pth"

BATCH_SIZE = 128


# =========================
# 2. Load MNIST test data
# =========================

transform = transforms.ToTensor()

test_dataset = datasets.MNIST(
    root=DATA_PATH,
    train=False,
    transform=transform,
    download=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# =========================
# 3. Load models
# =========================

softmax_model = SoftmaxClassifier().to(DEVICE)
cnn_model = CNN().to(DEVICE)

softmax_model.load_state_dict(
    torch.load(SOFTMAX_PATH, map_location=DEVICE)
)

cnn_model.load_state_dict(
    torch.load(CNN_PATH, map_location=DEVICE)
)

softmax_model.eval()
cnn_model.eval()


# =========================
# 4. Prediction function
# =========================

def get_predictions(model, loader):

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    return all_labels, all_predictions


# =========================
# 5. Get predictions
# =========================

print("Running Softmax prediction...")
softmax_labels, softmax_predictions = get_predictions(
    softmax_model,
    test_loader
)

print("Running CNN prediction...")
cnn_labels, cnn_predictions = get_predictions(
    cnn_model,
    test_loader
)


# =========================
# 6. Create confusion matrix
# =========================

def create_confusion_matrix(labels, predictions):

    matrix = torch.zeros(10, 10, dtype=torch.int64)

    for true_label, predicted_label in zip(labels, predictions):

        matrix[true_label, predicted_label] += 1

    return matrix.numpy()


softmax_cm = create_confusion_matrix(
    softmax_labels,
    softmax_predictions
)

cnn_cm = create_confusion_matrix(
    cnn_labels,
    cnn_predictions
)


# =========================
# 7. Calculate accuracy
# =========================

softmax_accuracy = (
    sum(
        true == pred
        for true, pred in zip(
            softmax_labels,
            softmax_predictions
        )
    )
    / len(softmax_labels)
) * 100


cnn_accuracy = (
    sum(
        true == pred
        for true, pred in zip(
            cnn_labels,
            cnn_predictions
        )
    )
    / len(cnn_labels)
) * 100


print()
print(f"Softmax Test Accuracy: {softmax_accuracy:.2f}%")
print(f"CNN Test Accuracy:     {cnn_accuracy:.2f}%")


# =========================
# 8. Plot confusion matrices
# =========================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

classes = list(range(10))


# ----- Softmax -----

axes[0].imshow(softmax_cm)

axes[0].set_title(
    f"Softmax Confusion Matrix\n"
    f"Accuracy: {softmax_accuracy:.2f}%"
)

axes[0].set_xlabel("Predicted Label")
axes[0].set_ylabel("True Label")

axes[0].set_xticks(classes)
axes[0].set_yticks(classes)

for i in range(10):
    for j in range(10):

        axes[0].text(
            j,
            i,
            softmax_cm[i, j],
            ha="center",
            va="center"
        )


# ----- CNN -----

axes[1].imshow(cnn_cm)

axes[1].set_title(
    f"CNN Confusion Matrix\n"
    f"Accuracy: {cnn_accuracy:.2f}%"
)

axes[1].set_xlabel("Predicted Label")
axes[1].set_ylabel("True Label")

axes[1].set_xticks(classes)
axes[1].set_yticks(classes)

for i in range(10):
    for j in range(10):

        axes[1].text(
            j,
            i,
            cnn_cm[i, j],
            ha="center",
            va="center"
        )


plt.tight_layout()

plt.savefig(
    "confusion_matrix_comparison.png",
    dpi=300
)

plt.show()