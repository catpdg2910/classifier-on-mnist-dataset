# khai báo thư viện
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch as torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader

# Tải data mnist dùng datasets của Torch
# train_data = datasets.MNIST(root = './MNIST DATASET', train = True, download = True, transform = transforms.ToTensor())
# test_data = datasets.MNIST(root = './MNIST DATASET', train = False, download = True, transform = transforms.ToTensor())

# train_loader = DataLoader(dataset = train_data, batch_size = 32, shuffle = True)
# test_loader = DataLoader(dataset = test_data, batch_size = 1, shuffle = False)

# Xây dựng mô hình
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size = 3, stride = 1, padding = 1)
        
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        
        self.linear1 = nn.Linear(7*7*16, 64)
        self.linear2 = nn.Linear(64, 10)
        
    def forward(self, X):
        X = self.maxpool1(F.relu(self.conv1(X)))
        X = self.maxpool2(F.relu(self.conv2(X)))
        X = torch.flatten(X, start_dim = 1)
        X = F.relu(self.linear1(X))
        X = self.linear2(X)
        return X
    
# def fit(model, Train_Loader, num_epoch):
#     optimizer = torch.optim.Adam(model.parameters())
#     loss_func = nn.CrossEntropyLoss()
#     model.train()
    
#     for epoch in range(num_epoch):
#         epoch_loss = 0
#         batch_num = 0
#         for X_batch, y_batch in Train_Loader:
#             optimizer.zero_grad()
#             y = model(X_batch)
#             loss = loss_func(y, y_batch)
#             loss.backward()
#             optimizer.step()
            
#             epoch_loss += loss
#             batch_num +=1
#         print(f'Average loss after epoch {epoch}: {epoch_loss/batch_num:.4f}')
        
# def eval(model, Test_Loader):
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for X_batch, y_batch in Test_Loader:
#             X_batch, y_batch = X_batch, y_batch
#             y_predict = model(X_batch)
#             prediction = torch.argmax(y_predict, dim=1)
            
#             correct += (prediction == y_batch).sum().item()
#             total += y_batch.size(0)
            
#     accuracy = (correct / total) * 100
#     print(f'Accuracy = {accuracy:.3f}%')

# # chạy thử 
# model = CNN()
# fit(model, train_loader, num_epoch=10)
# eval(model, test_loader)