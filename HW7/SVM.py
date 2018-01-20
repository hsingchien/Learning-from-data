import random as rand
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

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
	# while any(supervised_sign != w0_sign):
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

clf = svm.SVC(C = np.inf, kernel='linear')
r = 0
n_supp = 0
pr_pla = np.empty(1000)
pr_svm = np.empty(1000)
for i in range(0,1000):
	x = xspace()
	training_set = training_set_generate(100)
	while np.all(np.sign(x.test(training_set)) == 1) or np.all(np.sign(x.test(training_set)) == -1): 
		x = xspace()
		training_set = training_set_generate(10)
	#PLA
	w0 = np.array([0,0,0])
	w = PLA_train(10000,x,training_set,w0)
	w0 = w[0]
	w = x.w
	pr0 = error_prob(w0,w,10000)
	#SVM
	X = training_set[:,[1,2]]
	y = x.test(training_set)
	clf.fit(X,y)
	print(clf.coef_)
	w1 = np.array(clf.coef_[0])
	b = clf.intercept_[0]
	w1 = np.insert(w1,0,b)
	pr1 = error_prob(w1, w, 10000)
	pr_pla[i] = pr0
	pr_svm[i] = pr1
	# x_ax = np.arange(1,1001)	
	# n_supp += np.sum(clf.n_support_)
	plt.ion()
	plt.scatter(i, pr0,color='blue',s=4)
	plt.scatter(i, pr1, color='red',s=4)
	plt.ylim(0,1)
	# plt.scatter(i, n_supp/(i+1), color = 'green', s=4)
	plt.pause(0.0001)
	if pr1 < pr0:
		r += 1
print(r/1000)




# x = xspace()
# training_set = training_set_generate(10)
# w0 = np.array([0,0,0])
# w = PLA_train(10000,x,training_set,w0)
# w0 = w[0]
# w = x.w
# pr0 = error_prob(w0,w,10000)

# s = plot_w(w,w0,training_set)
# plt.show(s)

# SVM
# X = training_set[:,[1,2]]
# y = x.test(training_set)
# clf = svm.SVC(C = np.inf, kernel='linear')
# clf.fit(X,y)
# w = np.array(clf.coef_[0])
# b = clf.intercept_[0]
# w = np.insert(w,0,b)
# s=plot_w(w,x.w,training_set)
# plt.show()