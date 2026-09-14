#tải thư viện cần thiết
import matplotlib.pyplot as plt
import torchvision as tv
import numpy as np

#download data + chuyển mỗi ảnh sang tensor (1, 28, 28)
mnist_data = tv.datasets.MNIST(root = './MNIST DATASET', download = True, train = True, transform = tv.transforms.ToTensor())

#viết hàm softmax + cross - entropy loss
def softmax(x, axis = -1):
    max_z = np.max(x, axis = -1, keepdims = True)
    exp_z = np.exp(x - max_z)
    return exp_z/np.sum(exp_z, axis = axis, keepdims = True)
   
def cross_entropy_loss(predict, actual):
    m = actual.shape[0]
    log_p = -np.log(predict[range(m), actual]) # dễ hiểu nhưng chậm hơn: log_p = -np.log(predict[i, actual[i]]) for i in range(m)
    loss = np.sum(log_p)/m
    return loss

class SoftmaxClassification:
    def __init__(self, learning_rate = 0.01, num_classes = 10, num_features = 784):
        self.learning_rate = learning_rate
        self.weight = np.random.randn(num_features, num_classes)
        self.bias = np.zeros(num_classes)
        
    def train(self, X, y, epoch = 10000):
        #tính tới
        logits = np.dot(X, self.weight) + self.bias
        prob = softmax(logits)
        
        #backprobagation
        
        
        
        
        

# test softmax + cel
# a = np.array([[-1,-2,3,-2],[0,-2,1,6]])
# b = np.array([2,3])
# print(softmax(a))
# print(cross_entropy_loss(softmax(a), b))

# test vẽ hình
# image1, label1 = mnist_data[1]
# image2, label2 = mnist_data[2]
# image3, label3 = mnist_data[3]
# image4, label4 = mnist_data[4]
# fig, axs = plt.subplots(2,2, figsize = (10,10))
# axs[0,0].imshow(image1, cmap = 'gray')
# axs[0,0].set_title(f'Label: {label1}')
# axs[0,1].imshow(image2, cmap = 'gray')
# axs[0,1].set_title(f'Label: {label2}')
# axs[1,0].imshow(image3, cmap = 'gray')
# axs[1,0].set_title(f'Label: {label3}')
# axs[1,1].imshow(image4, cmap = 'gray')
# axs[1,1].set_title(f'Label: {label4}')
# plt.show()
# tensor, label = mnist_data[0]
# print(f'Tensor shape: {tensor.shape}')
# print(f'Label: {label}')
# print(f'Tensor values: {tensor}')
# print(tensor[0, 5, 21])
# plt.imshow(tensor.squeeze(), cmap = 'gray')
# plt.show()