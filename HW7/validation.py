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
x_in_train = x_in.iloc[0:25,:]
y_in_train = y_in.iloc[0:25]
x_in_val = x_in.iloc[25:,:]
y_in_val = y_in.iloc[25:]
x_out = data_out_trans.loc[:,data_out_trans.columns!='y']
y_out = data_out_trans.loc[:,'y']
def linear_sol(x_in, y_in):
	sudo_inv_x = np.linalg.inv(np.transpose(x_in).dot(x_in)).dot(np.transpose(x_in))
	weight = sudo_inv_x.dot(y_in)
	return weight
def error_est(x_in, y_in, weight):
	y_in_pred = np.sign(x_in.dot(weight))
	return np.mean(y_in != y_in_pred)
def linear_sol_constrain(x_in, y_in, lamb):
	sudo_inv_x_reg = np.linalg.inv(np.transpose(x_in).dot(x_in) + lamb * np.identity(np.shape(x_in)[1])).dot(np.transpose(x_in))
	weight_reg = sudo_inv_x_reg.dot(y_in)
	return weight_reg

k_list = [3,4,5,6,7]
errors = []
for k in k_list:
	ind = list(range(0,k))
	ind.append(7)
	weight = linear_sol(x_in_val.iloc[:,ind],y_in_val)
	err = error_est(x_out.iloc[:,ind],y_out,weight)
	errors.append(err)

print(errors)
plt.plot(k_list,errors)
plt.show()
