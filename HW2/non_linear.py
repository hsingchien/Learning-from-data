import random as rand
import math
import matplotlib.pyplot as plt
import numpy as np
def training_set_generate(n):
	rand.seed(11)
	w_one = np.ones([n,1])
	rand.seed(11)
	training_set = np.random.uniform(-1.0,1.0,[n,2])
	training_set = np.concatenate((w_one, training_set),1) #trainingset is [1, x1, x2; 1, x1, x2 ....]
	return training_set
def training_set_transform(x): #input a training set and do non_linear transform
	ones = x[:,0, None]
	x1 = x[:,1, None]
	x2 = x[:,2, None]
	x_transformed = np.concatenate((ones, x1, x2, x1*x2, x1**2, x2**2), axis = 1)
	return x_transformed
def signs(training_set):	
	training_set = training_set**2
	n = np.shape(training_set)[0]
	s_signs = np.sum(training_set, axis = 1)
	s_signs = np.sign(s_signs - 1.6) # 1 + x1^2 + x2^2 - 1.6
	ind = np.random.random_integers(0, n-1, math.ceil(n*0.1)) # 10% noise
	s_signs[ind] = s_signs[ind]*-1 # flip signs
	return s_signs
def error_in_sampe(w,y,training_set): # w: result of linear reg, y: training_set output, 10% noise
	pred = np.sign(training_set.dot(w))
	e = pred != y
	return sum(e)/np.size(y)
def error_out_sampe(w):
	x = training_set_generate(1000)
	x_t = training_set_transform(x)
	y = signs(x)
	p = np.sign(x_t.dot(w)) != y
	return sum(p)/1000


def li_reg(x,y):
	sudo_inv = np.linalg.inv(np.transpose(x).dot(x)).dot(np.transpose(x))
	w_linear = sudo_inv.dot(y)
	return w_linear
# question 8
# e = np.empty(1000)
# for i in range(0,1000):
# 	x = training_set_generate(1000)
# 	x_t = training_set_transform(x)
# 	y = signs(x)
# 	w_linear = li_reg(x_t,y)
# 	in_sapme_error = error_in_sampe(w_linear,y,x_t)
# 	e[i] = in_sapme_error
# print(np.mean(e))

# question 9
w = np.empty([100,6])
for i in range(0,100):
	x = training_set_generate(1000)
	x_t = training_set_transform(x)
	y = signs(x)
	w_linear = li_reg(x_t,y)
	w[i,:] = w_linear
w_m = np.mean(w, axis=0)
print(w_m)


# question 10
# e = np.empty(1000)
# for i in range(0,1000):
# 	e[i] = error_out_sampe(w_m)
# print(np.mean(e))

y_t = np.sign(x_t.dot(w_m))
x1 = x[:,1]
x2 = x[:,2]
plt.scatter(x1[y == 1], x2[y == 1], color = 'g')
plt.scatter(x1[y == -1], x2[y == -1], color = 'r')
# plt.scatter(x1[y_t != y], x2[y_t != y], color='b')
# plt.scatter(x1[y_t == y], x2[y_t == y], color='y')
plt.show()
