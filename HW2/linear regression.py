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
	def test(self, testee): #testee is [1,x1,x2]'
		output = testee.dot(self.w)
		return np.sign(output)

def training_set_generate(n):
	w_one = np.ones([n,1])
	training_set = np.random.uniform(-1.0,1.0,[n,2])
	training_set = np.concatenate((w_one, training_set),1) #trainingset is [1, x1, x2; 1, x1, x2 ....]
	return training_set

def error_prob_in_sample(w1,w2,training_set):
	pred1 = np.sign(training_set.dot(w1))
	pred2 = np.sign(training_set.dot(w2))
	p = pred1 != pred2
	return sum(p)/np.shape(training_set)[0]

def error_prob_out(w1,w2,n):
	w_one = np.ones([n,1])
	point_set = np.random.uniform(-1.0, 1.0, [n,2]) 
	point_set = np.concatenate((w_one, point_set),1)
	pred1 = np.sign(point_set.dot(w1))
	pred2 = np.sign(point_set.dot(w2))
	p = pred1 != pred2
	return sum(p)/n
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

#Question 5,6
# errors = np.empty([1000,2])
# for i in range(0,1000):
# 	t = xspace() #target function
# 	training_set = training_set_generate(100)
# 	w = t.w # [b,k,-1]
# 	y = np.sign(training_set.dot(w)) #y fore regression
# 	sudo_inverse = np.linalg.inv((np.transpose(training_set).dot(training_set))).dot(np.transpose(training_set))
# 	g =  sudo_inverse.dot(y)
# 	ein = error_prob_in_sample(w, g, training_set)
# 	eout = error_prob_out(w, g, 1000)
# 	errors[i, 0] = ein
# 	errors[i, 1] = eout
# print(np.mean(errors,axis=0))
 
#Question 7
iteration_times = np.empty(1000)
for i in range(0,1000):
	t = xspace() #target function
	training_set = training_set_generate(10)
	w = t.w # [b,k,-1]
	y = np.sign(training_set.dot(w)) #y fore regression
	sudo_inverse = np.linalg.inv((np.transpose(training_set).dot(training_set))).dot(np.transpose(training_set))
	g =  sudo_inverse.dot(y)
	p_out = PLA_train(1000, t, training_set, g)
	iteration_times[i] = p_out[1]
print(np.mean(iteration_times))

#plot
# x_train = training_set[:,1]
# y_train = training_set[:,2]
# x_ax = np.arange(-1,1,0.1)
# y_x = w[0]+w[1]*x_ax
# y_line = (g[0]+g[1]*x_ax)/-(g[2])
# plt.plot(x_ax,y_line, color = 'g')
# plt.plot(x_ax,y_x, color = 'k')
# plt.scatter(x_train,y_train)
# plt.xlim(-1,1)
# plt.ylim(-1,1)
# plt.show()

