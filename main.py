from SoftmaxClassification import SoftmaxClassifier
import torchvision as tv
import numpy as np
from torch.utils.data import DataLoader

train_dataset = tv.datasets.MNIST(root = './MNIST DATASET', train = True, transform = tv.transforms.ToTensor(), download = True)
test_dataset = tv.datasets.MNIST(root = './MNIST DATASET', train = False, transform = tv.transforms.ToTensor(), download = True)

train_loader = DataLoader(dataset = train_dataset, batch_size = 32, shuffle = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = 10000, shuffle = False)

def get_numpy_batches(dataLoader):
    for X_tensor, y_tensor in dataLoader:
        X_flat = X_tensor.view(X_tensor.shape[0], -1)
        yield X_flat.numpy(), y_tensor.numpy()
        
model = SoftmaxClassifier(0.01, 10, 784)

# for X_batch, y_batch in get_numpy_batches(train_loader):
#     model.train(X_batch, y_batch, True)
    
for epoch in range(30):
    epoch_loss = 0
    num_batches = 0
    for X_batch, y_batch in get_numpy_batches(train_loader):
        batch_loss = model.train(X_batch, y_batch)
        epoch_loss += batch_loss
        num_batches += 1
    ave_loss = epoch_loss/num_batches
    print(f'Hoan thanh epoch so: {epoch} - average loss = {ave_loss:.4f}')
    
    
#check thử
X_test_batch, y_test_batch = next(iter(get_numpy_batches(test_loader)))
predictions = model.predict(X_test_batch)
print('nhãn thực tế: ', y_test_batch)
print('nhãn dự đoán: ', predictions)

accuracy = np.mean(predictions == y_test_batch)*100
print(f"Độ chính xác trên batch test đầu tiên: {accuracy:.3f}%")



        