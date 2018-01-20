import numpy as np
n1 = np.random.random_sample(1000)
n2 = np.random.random_sample(1000)
n3 = np.concatenate((n1,n2),axis=0)
n3.shape = [2,1000]

n3 = np.min(n3,axis=0)
print(np.mean(n3))