import numpy as np
a = np.array([[[1]],[[1]]])
b = np.squeeze(a, 1)
print(a.shape)
print (b.shape)