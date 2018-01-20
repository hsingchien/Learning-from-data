import matplotlib.pyplot as pyplot
import numpy as np

def growth(N):
	return np.power(N,50)

x = np.arange(3,11,dtype = float)
x2 = np.arange(3,11,dtype=float)
y1 = np.sqrt(8/x*np.log((4*growth(2*x))/0.05))
y2 = np.sqrt(2*np.log(2*x*growth(x))/x)+np.sqrt(2/x*np.log(1/0.05))+1/x
y3 = (2/x+np.sqrt(4/np.square(x)+4/x*np.log(6*growth(2*x)/0.05)))/2
y4 = (2/x2+np.sqrt(4/np.square(x2)+4*(1-2/x2)*(np.log(4/0.05)+100*np.log(x2))/(2*x2)))/(2-4/x2)
pyplot.plot(x,y1, color = 'y')
pyplot.plot(x,y2, color = 'b')
pyplot.plot(x,y3, color = 'r')
pyplot.plot(x2,y4, color = 'g')
pyplot.show()