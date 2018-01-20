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

def Logist_NSGD(xsp,training_set,wt,step,terminate):
	n = training_set.shape[0]
	w=[wt]
	err=1
	w0 = x.w
	x_train = training_set[:,1]
	y_train = training_set[:,2]
	x_ax = np.arange(-1,1.1,0.1)
	y_line = (w0[0]+w0[1]*x_ax)/-(w0[2])
	s = plt.figure(1)
	plt.ion()
	plt.subplot(311)
	plt.axis([-1,1,-1,1])
	plt.plot(x_ax,y_line, color = 'g')
	plt.scatter(x_train,y_train, s=4)

	while(True): 
		if(len(w) == 1):
			pass
		elif(np.linalg.norm(w[-1]-w[-2]) < terminate):
			break
		y = xsp.test(training_set)
		yt = training_set.dot(wt)
		err = np.mean(np.log(1 + np.exp(-y * yt)))  # non sgd
		g_err = -np.mean(((training_set.T*y)/(1+np.exp(y*yt))).T, 0)
		wt = wt - step*g_err
		y_x = (wt[0]+wt[1]*x_ax)/-(wt[2])
		plt.figure(1)
		plt.subplot(311)
		plt.plot(x_ax,y_x,color = 'k',alpha = 0.1)
		plt.subplot(312)
		# plt.axis([0,1000,0,3])
		plt.scatter(len(w),err, s = 4)
		w.append(wt)
		plt.subplot(313)
		plt.scatter(len(w), np.linalg.norm(w[-1]-w[-2]), s =4)
		plt.pause(0.0001)

	return [w, s]

x = xspace()
training_set = training_set_generate(100)
w0 = np.array([0,0,0])
step = 0.1
terminate = 0.01
w = Logist_NSGD(x,training_set,w0,step,terminate)
plt.show(w[1])
