import random as rand
import numpy as np
import matplotlib.pyplot as plt
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
	def test(self, testee): #testee is [1,xi,yi]', return the right label
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
	while i < n:
		if (supervised_sign == w0_sign).all():
			break
		for j in range(0,len(supervised_sign)):
			sup = supervised_sign[j]
			ws = w0_sign[j]
			if sup != ws:
				w0 = w0 + sup*training_set[j,:]
				w0_sign = training_set.dot(w0)
				w0_sign = np.sign(w0_sign)
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