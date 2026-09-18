import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch as torch

train_data = datasets.MNIST(root = './MNIST DATASET', train = True, transform = transforms.ToTensor(), download = True)
test_data = datasets.MNIST(root = './MNIST DATASET', train = False, transform = transforms.ToTensor(), download = True)

train_loader = DataLoader(dataset = train_data, batch_size = 32, shuffle = True)
test_loader = DataLoader(dataset = test_data, batch_size = 10000, shuffle = False)

model = nn.Sequential(
    nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size = 2, stride = 2),
    
    nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size = 2, stride = 2),
    
    nn.Flatten(),
    nn.Linear(in_features = 7*7*64, out_features = 128),
    nn.ReLU(),
    nn.Linear(in_features = 128, out_features = 10)    
)

optimizer = optim.Adam(params = model.parameters())
loss_func = nn.CrossEntropyLoss()

epochs = 20
for epoch in range(epochs):
    epoch_loss = 0
    num_epoch = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outs = model(X_batch)
        loss = loss_func(outs, y_batch)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss
        num_epoch += 1
    print(f"average loss after epoch {epoch}: {epoch_loss/num_epoch:.4f}")
    
X_test, y_test = next(iter(test_loader))
with torch.no_grad():
    y_predict = model(X_test)
    prediction = torch.argmax(y_predict, 1, keepdim = False)
    
print('Nhãn Thực tế: ', y_test)
print('Nhãn dự đoán: ', prediction)

correct = (y_test == prediction).sum().item()
accuracy = correct/len(y_test)*100

print('do chinh xac: ', accuracy)

