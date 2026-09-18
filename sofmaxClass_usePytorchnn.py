import torch as torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision as tv
from torch.utils.data import DataLoader
import torch.optim as optim

train_data = datasets.MNIST(root = './MNIST DATASET', train = True, transform = tv.transforms.ToTensor(), download = True)
test_data = datasets.MNIST(root = './MNIST DATASET', train = False, transform = tv.transforms.ToTensor(), download = True)

train_loader = DataLoader(dataset = train_data, batch_size = 32, shuffle = True)
test_loader = DataLoader(dataset = test_data, batch_size = 10000, shuffle = False)

# set up model + chọn hàm loss: CEL + chọn optimizer: SGD/ADAM
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_features = 784, out_features = 10)
)

loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)

epochs = 20
for epoch in range(epochs):
    epoch_loss = 0
    num_batches = 0
    for X_batch, y_batch in train_loader:
        # xóa grad của batch trước
        optimizer.zero_grad()
        
        # forward
        outs = model(X_batch)
         
        # tính loss
        loss = loss_func(outs, y_batch)
        
        # tính gradient của loss
        loss.backward()
        
        # cập nhật weight + bias
        optimizer.step()
        
        # tính loss sau mỗi epoch
        epoch_loss += loss
        num_batches += 1
        
    ave_loss = epoch_loss/num_batches
    print(f"Average Loss after epoch {epoch}: {ave_loss:.4f}")
    
# chạy trên tập test
X_test, y_test = next(iter(test_loader))

with torch.no_grad():
    y_predict = model(X_test)
    prediction = torch.argmax(y_predict, 1, False)
    
print('Nhãn Thực tế: ', y_test)
print('Nhãn dự đoán: ', prediction)

correct = (y_test == prediction).sum().item()
accuracy = correct/len(y_test)*100

print('do chinh xac: ', accuracy)





