import numpy as np

a = np.array([[1, 1, 1],[2, 2, 2],[3,3,3]])
b = np.array([[1, 2],[1, 2]])
def convolution(X, kernel):
    k_size = kernel.shape[0]
    X_dai = X.shape[1]
    X_rong = X.shape[0]
    res = np.zeros((X_rong - k_size + 1, X_dai - k_size + 1))
    for i in range(X_rong - k_size + 1):
        for j in range(X_dai - k_size + 1):
            res[i,j] = np.sum(X[i:i+k_size, j:j+k_size]*kernel)
    print(res)
    
convolution(a, b)