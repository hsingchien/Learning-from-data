import pandas as pd
from sklearn import svm
import numpy as np
from sklearn.model_selection import KFold


train = pd.read_excel('train.xlsx')
test = pd.read_excel('test.xlsx')
train.columns = ['dig','inten','symm']
test.columns = ['dig','inten','symm']


def one_vs_all(tar, train, Co, Q):
	Y = np.zeros(train.shape[0])
	Y[train_c.dig == tar] = 1 # Y is the label
	X = train.iloc[0:, 1:3] # X is the input
	gram = (1+np.dot(X, X.T))**Q ## pre-compute the kernel
	clf = svm.SVC(C = Co, kernel = "precomputed")
	clf.fit(gram, Y)
	return [clf,gram,Y]
def one_vs_one(tar1, tar2, train, Co, Q):
	X = train[train['dig'].isin([tar1,tar2])]
	Y = np.zeros(X.shape[0])
	Y[X.dig == tar1] = 1 # set up Y
	X = X.iloc[0:, 1:3]
	gram = (1+np.dot(X, X.T))**Q ## pre-compute the kernel
	clf = svm.SVC(C = Co, kernel = "precomputed")
	clf.fit(gram, Y)
	return clf
def error_compute(clf, train, test,Q):
	Xout = test[test['dig'].isin([1,5])]
	Yout = np.zeros(Xout.shape[0])
	Yout[Xout.dig == 1] = 1
	Xout = Xout.iloc[:,1:3]
	X = train[train['dig'].isin([1,5])]
	X = X.iloc[0:,1:3]
	gram_out = (1+np.dot(Xout, X.T))**Q
	test_pred = clf.predict(gram_out)
	error_out = 1-np.sum(np.sign(test_pred) == np.sign(Yout))/test_pred.shape[0]
	return error_out
def cross_val(train, tar1, tar2, k, c, Q):
	# step1 k fold split
	kf = KFold(n_splits = k, shuffle = True)
	X_use = train[train['dig'].isin([tar1,tar2])]
	error = np.empty(k)
	i = 0
	for train_index, test_index in kf.split(X_use): 
	# X_use is dig(tar1 or tar2), par1, par2
		Xtrain = X_use.iloc[train_index,:]
		Xtest = X_use.iloc[test_index,:]
		clf = one_vs_one(tar1,tar2,Xtrain,c,Q)
		err = error_compute(clf,Xtrain,Xtest,Q)
		error[i] = err
		i += 1
	return np.mean(err)


train = pd.read_excel('train.xlsx')
test = pd.read_excel('test.xlsx')
train.columns = ['dig','inten','symm']
test.columns = ['dig','inten','symm']
s = []
error = np.empty(100)
Cs = [0.0001,0.001,0.01,0.1,1]
for i in range(0,100):
	err = cross_val(train, 1, 5, 10, 0.001, 2)
	error[i] = err
print(np.mean(error))