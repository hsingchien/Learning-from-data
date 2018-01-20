import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
def n_l_transform(data_in):
	data_in_trans = data_in.assign(x3 = np.power(data_in['x1'],2))
	data_in_trans = data_in_trans.assign(x4 = np.power(data_in['x2'],2))
	data_in_trans = data_in_trans.assign(x5 = data_in['x1']*data_in['x2'])
	data_in_trans = data_in_trans.assign(x6 = np.absolute(data_in['x1']-data_in['x2']))
	data_in_trans = data_in_trans.assign(x7 = np.absolute(data_in['x1']+data_in['x2']))
	data_in_trans['x0'] = 1
	return data_in_trans
data_in = pd.read_csv('in.csv')
data_out = pd.read_csv('out.csv')
data_in_trans = n_l_transform(data_in)
data_out_trans = n_l_transform(data_out)
x_in = data_in_trans.loc[:,data_in_trans.columns!='y']
y_in = data_in_trans.loc[:,'y']
x_out = data_out_trans.loc[:,data_out_trans.columns!='y']
y_out = data_out_trans.loc[:,'y']
def linear_sol(x_in, y_in):
	sudo_inv_x = np.linalg.inv(np.transpose(x_in).dot(x_in)).dot(np.transpose(x_in))
	weight = sudo_inv_x.dot(y_in)
	return weight
def error_est(x_in, y_in, weight):
	y_in_pred = np.sign(x_in.dot(weight))
	return np.mean(y_in != y_in_pred)

weight = linear_sol(x_in,y_in)
err_in = error_est(x_in, y_in, weight)
err_out = error_est(x_out, y_out, weight)
print('Q2', 'in-sample error:', err_in,'out-of sample error:', err_out)
# data_in.loc['y']
s = plt.figure(1)
plt.scatter(data_out[data_out.y > 0].loc[:,'x1'], data_out[data_out.y>0].loc[:,'x2'], c='green')
plt.scatter(data_out[data_out.y < 0].loc[:,'x1'], data_out[data_out.y<0].loc[:,'x2'], c='red')
x1 = np.linspace(-1,1,100)
x2 = np.linspace(-1,1,100)
X1,X2 = np.meshgrid(x1,x2)
F = weight[0]*X1+weight[1]*X2+weight[2]*X1**2+weight[3]*X2**2+weight[4]*X1*X2+weight[5]*np.absolute(X1-X2)+weight[6]*np.absolute(X1+X2)+weight[7]
plt.contour(X1,X2,F,[0])
def linear_sol_constrain(x_in, y_in, lamb):
	sudo_inv_x_reg = np.linalg.inv(np.transpose(x_in).dot(x_in) + np.power(10.,lamb) * np.identity(np.shape(x_in)[1])).dot(np.transpose(x_in))
	weight_reg = sudo_inv_x_reg.dot(y_in)
	return weight_reg

lamb = -3
weight_reg = linear_sol_constrain(x_in,y_in,lamb)
err_in_reg = error_est(x_in,y_in,weight_reg)
err_out_reg = error_est(x_out,y_out,weight_reg)
print('Q3', 'in-sample reg err:', err_in_reg, 'out-of-sample reg err:', err_out_reg)
err = []
lamlist = [2,1,0,-1,-2]
for lamb in lamlist:
	weight_reg = linear_sol_constrain(x_in,y_in,lamb)
	err.append(error_est(x_out,y_out,weight_reg))
# plt.plot(lamlist,err)
# plt.show()
err_data = pd.DataFrame(err,index=lamlist)
err_data.plot()
print(err_data.loc[-1])
plt.show()