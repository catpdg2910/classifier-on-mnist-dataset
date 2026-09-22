import torch
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from models.cnn_module import CNN
from models.SoftmaxClassification import SoftmaxClassifier

# ==========================================
# Configuration
# ==========================================

MODEL_PATH = "./checkpoints/softmax.pth"

NUM_IMAGES = 9

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ==========================================
# Load MNIST test dataset
# ==========================================

transform = transforms.ToTensor()

test_dataset = datasets.MNIST(
    root="./MNIST DATASET",
    train=False,
    transform=transform,
    download=False
)

print("Test samples:", len(test_dataset))


# ==========================================
# Create model
# ==========================================

model = SoftmaxClassifier()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.to(device)
model.eval()


# ==========================================
# Select 9 random images
# ==========================================

# Dùng seed 42 để lần nào chạy cũng
# chọn cùng 9 ảnh
torch.manual_seed(42)

indices = torch.randperm(
    len(test_dataset)
)[:NUM_IMAGES]


# ==========================================
# Predict and display
# ==========================================

fig, axes = plt.subplots(
    3,
    3,
    figsize=(8, 8)
)

with torch.no_grad():

    for ax, index in zip(axes.flat, indices):

        image, true_label = test_dataset[index]

        # Thêm batch dimension
        image_input = image.unsqueeze(0).to(device)

        # Prediction
        output = model(image_input)

        predicted_label = torch.argmax(
            output,
            dim=1
        ).item()

        # Display image
        ax.imshow(
            image.squeeze(),
            cmap="gray"
        )

        ax.set_title(
            f"True: {true_label} | Pred: {predicted_label}",
            fontsize=11
        )

        ax.axis("off")


plt.suptitle(
    "softmax - Random MNIST Test Predictions",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    "random_predictions.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()