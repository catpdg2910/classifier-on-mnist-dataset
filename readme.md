# MNIST Classifier

This project implements and compares different neural network models for handwritten digit classification on the MNIST dataset.

## Models

The project includes three different models:

* **Softmax Classifier** – a simple linear classification model.
* **CNN** – a Convolutional Neural Network for extracting spatial features from images.
* **ViT** – a Vision Transformer that applies the Transformer architecture to image patches.

## Dataset

The project uses the **MNIST handwritten digit dataset**, which contains grayscale images of digits from 0 to 9.

The dataset is divided into:

* Training set: 50,000 images
* Validation set: 10,000 images
* Test set: 10,000 images

## Experiments

The project focuses on comparing the models based on:

* Training and validation performance
* Test accuracy
* Learning curves
* Confusion matrices
* Prediction results on test images
* Effect of different learning rates

## Results

The models achieve different classification performance on the MNIST test set:

| Model   | Test Accuracy |
| ------- | ------------- |
| Softmax | 92.59%        |
| CNN     | 98.78%        |
| ViT     | 97.90%        |

The experiments provide a simple comparison between a linear model, a convolutional neural network, and a Vision Transformer on the same dataset.

## Project Structure

```text
MNIST-Classifier/
│
├── models/
│   ├── SoftmaxClassification.py
│   ├── cnn_module.py
│   └── ViT.py
│
├── utils/
│   ├── data.py
│   ├── train.py
│   └── plot.py
│
├── main.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Technologies

* Python
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* MNIST Dataset
