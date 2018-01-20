import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
import sklearn.cluster as clt

def targ_func(X): # X is numpy array organized as [x1,x2]
	y = np.sign(X[:,1] - X[:,0] + 0.25*np.sin(np.pi*X[:,0]))
	return y

def training_set_generate(n):
	training_set = np.random.uniform(-1.0,1.0,[n,2])
	return training_set

class RBF_regular:
	def __init__(self, K, gamma):
		self.K = K
		self.gamma = gamma
		self.w = None
	def fit(self, train, y):
		# step 1 find the centers
		K = self.K
		gamma = self.gamma
		clust = clt.KMeans(n_clusters = K)
		clust.fit(train)
		centers = clust.cluster_centers_
		# step2 build phi matrix
		phi = np.empty([np.shape(train)[0], K])
		for i in range(0, np.shape(phi)[0]):
			for j in range(0,K):
				phi[i,j] = np.exp(-gamma*np.sum((train[i,]-centers[j,])**2))
		# step3 calculate pseudo inverse
		pseudo = np.linalg.pinv(phi)
		w = np.dot(pseudo, y)
		self.weight = w
		self.centers = centers
	def predict(self, x):
		w = self.weight
		centers = self.centers
		gamma = self.gamma
		K = self.K
		h = 0
		for i in range(0,K):
			h += w[i] * np.exp(-gamma*np.sum((x-centers[i,])**2, axis = 1))
		return np.sign(h)
	def score(self, x, y):
		y_pred = self.predict(x)
		return metrics.accuracy_score(y_pred, y)



# Q13
# c = 0
# for i in range(0,10000):
# 	train = training_set_generate(100)
# 	y = targ_func(train)
# 	clf = svm.SVC(C=np.inf, gamma = 1.5, kernel = 'rbf')
# 	clf.fit(train, y)
# 	E_in = 1 - clf.score(train, y)
# 	if(E_in > 0):
# 		c += 1
# print(c/10000)
# Q14,15
# c = 0
# s = plt.figure(1)
# for i in range(0, 100):
# 	# generate data sets
# 	train = training_set_generate(100)
# 	test = training_set_generate(1000)
# 	y_in = targ_func(train)
# 	y_out = targ_func(test)
# 	# svm
# 	clf = svm.SVC(C=np.inf, gamma = 1.5, kernel = 'rbf')
# 	clf.fit(train, y_in)
# 	E_out_svm = 1 - clf.score(test, y_out)
# 	# rbf regular
# 	rbf = RBF_regular(9,1.5)
# 	rbf.fit(train, y_in)
# 	E_out_rbf = 1 - rbf.score(test, y_out)
# 	# count the outperform times
# 	if(E_out_rbf > E_out_svm):
# 		c += 1
# 	# plot
# 	plt.scatter(i,E_out_rbf,c='red')
# 	plt.scatter(i,E_out_svm,c='green')
# print(c/100)
# plt.show()

# Q16
# err = pd.DataFrame({'delta_ein': np.empty(100), 'delta_eout': np.empty(100)})
# for i in range(0,100):
# 	train = training_set_generate(100)
# 	test = training_set_generate(1000)
# 	y_in = targ_func(train)
# 	y_out = targ_func(test)
# 	## cluster
# 	rbf9 = RBF_regular(9,1.5)
# 	rbf12 = RBF_regular(12,1.5)
# 	rbf9.fit(train, y_in)
# 	rbf12.fit(train, y_in)
# 	e_in9 = 1-rbf9.score(train, y_in)
# 	e_in12 = 1-rbf12.score(train, y_in)
# 	e_out9 = 1-rbf9.score(test, y_out)
# 	e_out12 = 1-rbf12.score(test, y_out)
# 	err.iloc[i,0] = e_in12 - e_in9
# 	err.iloc[i,1] = e_out12 - e_out9
# err.plot()
# plt.show()

# Q17
# err = pd.DataFrame({'delta_ein': np.empty(100), 'delta_eout': np.empty(100)})
# for i in range(0,100):
# 	train = training_set_generate(100)
# 	test = training_set_generate(1000)
# 	y_in = targ_func(train)
# 	y_out = targ_func(test)
# 	## cluster
# 	rbf15 = RBF_regular(9,1.5)
# 	rbf2 = RBF_regular(9,2)
# 	rbf15.fit(train, y_in)
# 	rbf2.fit(train, y_in)
# 	e_in15 = 1-rbf15.score(train, y_in)
# 	e_in2 = 1-rbf2.score(train, y_in)
# 	e_out15 = 1-rbf15.score(test, y_out)
# 	e_out2 = 1-rbf2.score(test, y_out)
# 	err.iloc[i,0] = e_in2 - e_in15
# 	err.iloc[i,1] = e_out2 - e_out15
# print(err[err.delta_ein <= 0].shape[0]/100)	
# print(err[err.delta_eout <= 0].shape[0]/100)
# err.plot()
# plt.show()

# Q18
# c = 0
# for i in range(0, 1000):
# 	train = training_set_generate(100)
# 	y_in = targ_func(train)
# 	rbf = RBF_regular(9, 1.5)
# 	rbf.fit(train, y_in)
# 	score = rbf.score(train, y_in)
# 	if(score == 1):
# 		c += 1
# print(c/1000)

f = plt.figure(0)
train = training_set_generate(100)
y_in = targ_func(train)
plt.scatter(train[y_in == 1,0],train[y_in == 1,1], c = 'green')
plt.scatter(train[y_in == -1,0], train[y_in == -1,1], c= 'red')
rbf = RBF_regular(9, 1.5)
rbf.fit(train, y_in)

x1 = np.linspace(-1,1,1000)
x2 = np.linspace(-1,1,1000)
X1,X2 = np.meshgrid(x1,x2)
FX1 = np.ndarray.flatten(X1)
FX2 = np.ndarray.flatten(X2)
M = np.array((FX1,FX2))
F = np.reshape(rbf.predict(np.transpose(M)), np.shape(X1))
plt.contour(X1,X2,F,[0])
plt.show()