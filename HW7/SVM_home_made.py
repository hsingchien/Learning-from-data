import random as rand
import numpy as np
import matplotlib.pyplot as plt
import quadprog as qp
from nearestPD import *

class xspace:
	def __init__(self):
		rand.seed(9)
		x1 = rand.uniform(-1,1)
		x2 = rand.uniform(-1,1)
		y1 = rand.uniform(-1,1)
		y2 = rand.uniform(-1,1)
		k = (y1-y2)/(x1-x2)
		b = y1-k*x1
		self.w = np.array([b,k,-1])
	def test(self, testee): #testee is [1,xi,yi]'
		output = testee.dot(self.w)
		return np.sign(output)

def training_set_generate(n):
	w_one = np.ones([n,1])
	rand.seed(11)
	training_set = np.random.uniform(-1.0,1.0,[n,2])
	training_set = np.concatenate((w_one, training_set),1)
	return training_set

def PLA_train(n, x, training_set, w0): # n iteration cycle; x target x space; training set; w0, initial value
	supervised_sign = x.test(training_set) #the right answer
	#output the correct sign as a column array for reference
	w0_sign = training_set.dot(w0) #w0 sign
	w0_sign = np.sign(w0_sign) 
	i = 0
	while i < n and any(supervised_sign != w0_sign):
		unmatch = np.nonzero(w0_sign != supervised_sign)
		j = np.random.choice(unmatch[0])
		sup = supervised_sign[j]
		ws = w0_sign[j]
		w0 = w0 + sup*training_set[j,:]
		i += 1
	return [w0,i]

def error_prob(w1,w2,n):
	w_one = np.ones([n,1])
	point_set = np.random.uniform(-1.0, 1.0, [n,2]) 
	point_set = np.concatenate((w_one, point_set),1)
	pred1 = np.sign(point_set.dot(w1))
	pred2 = np.sign(point_set.dot(w2))
	p = pred1 != pred2
	return sum(p)/n

def plot_w(w1, w2, training_set):
	s = plt.figure(1)
	x_ax = np.arange(-1,1,0.1)
	y_ax = np.arange(-1,1,0.1)
	y1 = w1[0]/(-w1[2])+x_ax*w1[1]/(-w1[2])
	y2 = w2[0]/(-w2[2])+x_ax*w2[1]/(-w2[2])
	plt.plot(x_ax,y1,color='blue')
	plt.plot(x_ax,y2,color='yellow')
	s_y1 = np.sign(training_set.dot(w1))
	s_y2 = np.sign(training_set.dot(w2))
	plt.scatter(training_set[s_y1 == s_y2,1], training_set[s_y1 == s_y2,2],color = 'green')
	plt.scatter(training_set[s_y1 != s_y2,1], training_set[s_y1 != s_y2,2],color = 'red')
	plt.xlim(-1,1)
	plt.ylim(-1,1)
	return s



def svm_solve(training_set, y):
	X = training_set[:,1:] ## isolate X
	n = np.shape(training_set)[0]
	quad_coef = np.empty([n,n])
	for i in range(0,n):
		for j in range(0,n):
			quad_coef[i,j] = y[i]*y[j]*X[i,:].dot(X[j,:])
	y.shape = [n,1]
	quad_coef = nearestPD(quad_coef)
	alpha = qp.solve_qp(quad_coef,np.ones([n]),y,np.zeros([1]),n)[0]
	alpha.shape = [n,1]
	w = np.sum(alpha*y*X, axis=0)
	m = np.random.choice(np.nonzero(alpha[:,] > 0)[0])
	b = 1/y[m] - w.dot(X[m,])
	w = np.insert(w,0,b)
	return w



x = xspace()
training_set = training_set_generate(100)
# w0 = np.array([0,0,0])
# w = PLA_train(1000,x,training_set,w0)
# print(w[0], w[1])
# w0 = w[0]
# w = x.w
# print("w0", np.sign(training_set.dot(w0)))
# print("w", np.sign(training_set.dot(w)))
# pr = error_prob(w0,w,10000)
# print(pr)
# s = plot_w(w,w0,training_set)
# plt.show(s)

## SVM
y = x.test(training_set)
svm_res = svm_solve(training_set,y)
print(svm_res)
s = plot_w(svm_res,x.w,training_set)
plt.show()