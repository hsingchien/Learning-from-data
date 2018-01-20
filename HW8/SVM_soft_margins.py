import pandas as pd
from sklearn import svm
import numpy as np
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

err = []
sv = []
err_out = []
# for i in np.arange(0,10,2):
# 	fit = one_vs_all(i,train,0.01,2)
# 	clf = fit[0]
# 	Y = fit[2]
# 	gram = fit[1]
# 	train_pred = clf.predict(gram)
# 	# error = 1-np.sum(np.sign(train_pred) == np.sign(Y))/train_pred.shape[0]
# 	# err.append(error)
# 	sv.append(np.sum(clf.n_support_))
# print(sv)

def one_vs_one(tar1, tar2, train, Co, Q):
	X = train[train['dig'].isin([tar1,tar2])]
	Y = np.zeros(X.shape[0])
	Y[X.dig == tar1] = 1 # set up Y
	X = X.iloc[0:, 1:3]
	gram = (1+np.dot(X, X.T))**Q ## pre-compute the kernel
	clf = svm.SVC(C = Co, kernel = "precomputed")
	clf.fit(gram, Y)
	return [clf,gram,Y]
def error_compute(clf, train, test,Q):
	Xout = Xout = test[test['dig'].isin([1,5])]
	Yout = np.zeros(Xout.shape[0])
	Yout[Xout.dig == 1] = 1
	Xout = Xout.iloc[:,1:3]
	X = train[train['dig'].isin([1,5])]
	X = X.iloc[0:,1:3]
	gram_out = (1+np.dot(Xout, X.T))**Q
	test_pred = clf.predict(gram_out)
	error_out = 1-np.sum(np.sign(test_pred) == np.sign(Yout))/test_pred.shape[0]
	return error_out
# for c in [0.001,0.01,0.1,1]:
# 	fit = one_vs_one(1,5,train,c,2)
# 	clf = fit[0]
# 	error = error_compute(clf,train,train,2)
# 	err.append(error)
# 	sv.append(np.sum(clf.n_support_))
# 	# Eout compute
# 	error_out = error_compute(clf,train,test,2)
# 	err_out.append(error_out)
# print('in sample error',err)
# print('number of supporter vectors',sv)
# print('out of sample error', err_out)
error = np.empty([4,4])
sv = np.empty([4,2])
i = 0
for c in [0.0001,0.001,0.01,1]:
	fit2 = one_vs_one(1,5,train,c,2)
	fit5 = one_vs_one(1,5,train,c,5)
	clf2 = fit2[0]
	clf5 = fit5[0]
	sv[i,:] = np.array([np.sum(clf2.n_support_), np.sum(clf5.n_support_)])
	err2_in = error_compute(clf2, train, train,2)
	err5_in = error_compute(clf5, train, train,5)
	err2_out = error_compute(clf2, train, test,2)
	err5_out = error_compute(clf5, train, test,5)
	error[i,:] = np.array([err2_in,err2_out,err5_in,err5_out])
	i += 1
error = pd.DataFrame(error)
error.columns = ['Q=2 Ein', 'Q=2 Eout', 'Q=5 Ein', 'Q=5 Eout']
sv = pd.DataFrame(sv)
sv.columns = ['Q=2 n_sv','Q=5 n_sv']
print(error)
print(sv)


