import random as rand
import numpy as np
import matplotlib.pyplot as plt

class xspace:
	def __init__(self):
		rand.seed(3)
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
	rand.seed(2)
	training_set = np.random.uniform(-1.0,1.0,[n,2])
	training_set = np.concatenate((w_one, training_set),1)
	return training_set

def Logist_SGD(xsp,training_set,wt,step,terminate):
	n = training_set.shape[0]
	w=[wt]
	err=1
	w0 = x.w
	x_train = training_set[:,1]
	y_train = training_set[:,2]
	x_ax = np.arange(-1,1.1,0.1)
	y_line = (w0[0]+w0[1]*x_ax)/-(w0[2])
	s = plt.figure(1)
	# plt.ion()
	# plt.subplot(311)
	# plt.axis([-1,1,-1,1])
	# plt.plot(x_ax,y_line, color = 'g')
	# plt.scatter(x_train,y_train, s=4)
	index = np.arange(0,n,1)
	j = 0
	k = 0
	while(True): # loop epochs
		
		if(len(w) == 1):
			pass
		elif(np.linalg.norm(w[-1]-w[-2]) < terminate):
			# y_x = (wt[0]+wt[1]*x_ax)/-(wt[2])
			# plt.figure(1)
			# plt.subplot(311)
			# plt.plot(x_ax,y_x,color = 'r')
			break
		np.random.shuffle(index)
		for i in range(0,n): # start epoch
			k += 1
			rn = index[i]
			xrn = training_set[rn,:]
			yrn = xsp.test(xrn)
			err = np.log(1 + np.exp(-yrn * wt.dot(xrn)))	# sgd
			g_err = (-yrn*xrn)/(1+np.exp(yrn * wt.dot(xrn)))
			wt = wt - step*g_err
			# plt.subplot(312)
			# plt.scatter(k,err, s = 4)
			# end of epoch
		j += 1
		w.append(wt)
		# plt.figure(1)
		# plt.subplot(313)
		# plt.scatter(len(w), np.linalg.norm(w[-1]-w[-2]), s =4)
		# y_x = (wt[0]+wt[1]*x_ax)/-(wt[2])
		# plt.subplot(311)
		# plt.plot(x_ax,y_x,color = 'k',alpha = 0.1)
		# plt.pause(0.0001)
		
	return [w, s, j]

def Eout(w,h,n):
	test_set = training_set_generate(n)
	h_result = test_set.dot(h)
	w_result = np.sign(test_set.dot(w))
	# theta = lambda x: np.exp(x)/(1+np.exp(x))
	# p_h = theta(1 + h_result * -w_result)
	eout = np.mean(np.log(1 + np.exp(h_result * -(w_result))))
	return eout

eout = []
j = []
for i in range(0,100):
	x = xspace()
	training_set = training_set_generate(100)
	h0 = np.array([0,0,0])
	step = 0.01
	terminate = 0.01
	output = Logist_SGD(x,training_set,h0,step,terminate)
	h = output[0]
	h = h[-1]
	j.append(output[2])
	w = x.w
	eout.append(Eout(w,h,10000))

	# plt.show(output[1])	
print('Epochs\t', np.mean(j))
print('Eout\t', np.mean(eout))