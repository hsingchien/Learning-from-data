import pandas as pd
import numpy as np


def linear_sol_constrain(x_in, y_in, lamb):
	sudo_inv_x_reg = np.linalg.inv(np.transpose(x_in).dot(x_in) + lamb * np.identity(np.shape(x_in)[1])).dot(np.transpose(x_in))
	weight_reg = sudo_inv_x_reg.dot(y_in)
	return weight_reg

def one_vs_all(tar, train, lamb):
	Y = np.ones(train.shape[0])*-1 # set all to -1
	Y[train.dig == tar] = 1 # Y is the label
	X = train.iloc[0:, 1:] # X is the input
	ones = pd.DataFrame({'const': np.ones(train.shape[0])})
	X = pd.concat([ones, X], axis=1)
	w = linear_sol_constrain(X, Y, lamb)
	return [w, X, Y]
def one_vs_one(tar1, tar2, train, lamb):
	ones = pd.DataFrame({'const': np.ones(train.shape[0])})
	X = train.copy()
	X.insert(1,'const', ones)
	X = X[X['dig'].isin([tar1, tar2])]
	Y = -1*np.ones(X.shape[0])
	Y[X.dig == tar1] = 1 # set up Y
	X = X.iloc[0:, 1:]
	w = linear_sol_constrain(X, Y, lamb)
	return[w, X, Y]

def error_est(x_in, y_in, weight):
	y_in_pred = np.sign(x_in.dot(weight))
	return np.mean(y_in != y_in_pred)
def error_est_out(tar, test_nl, weight):
	X_out = test_nl.iloc[0:,1:]
	ones = ones = pd.DataFrame({'const': np.ones(test_nl.shape[0])})
	X_out.reset_index(drop = True, inplace = True)
	X_out = pd.concat([ones,X_out], axis = 1)
	Y_out = np.ones(test_nl.shape[0])*-1 # set all to -1
	Y_out[test_nl.dig == tar] = 1
	return error_est(X_out, Y_out, weight)

train = pd.read_excel('train.xlsx')
test = pd.read_excel('test.xlsx')
train.columns = ['dig','inten','symm']
test.columns = ['dig','inten','symm']



def n_l_transform(data_in):
	data_in_trans = data_in.assign(x3 = np.power(data_in['inten'],2))
	data_in_trans = data_in_trans.assign(x4 = np.power(data_in['symm'],2))
	data_in_trans = data_in_trans.assign(x5 = data_in['inten']*data_in['symm'])
	return data_in_trans

train_nl = n_l_transform(train)
test_nl = n_l_transform(test)
# err = []
# for i in range(0,10):
# 	sol = one_vs_all(i, train_nl, 1)
# 	w = sol[0]
# 	X_in = sol[1]
# 	Y_in = sol[2]
# 	err.append(error_est_out(i, test_nl, w))
# res = pd.DataFrame({'with trans': np.array(err)})
# err = []
# for i in range(0,10):
# 	sol = one_vs_all(i, train, 1)
# 	w = sol[0]
# 	X = sol[1]
# 	Y = sol[2]
# 	err.append(error_est_out(i, test, w))
# res = res.assign(without = np.array(err))
# res = res.assign(without_95 = np.array(err)*0.95)

# print(res)

sol_transform0 = one_vs_one(1, 5, train_nl, 0.01)
sol_transform1 = one_vs_one(1, 5, train_nl, 1)
err_1 = error_est(sol_transform0[1], sol_transform0[2], sol_transform0[0])
err_2 = error_est(sol_transform1[1], sol_transform1[2], sol_transform1[0])
test_use = test_nl[test_nl['dig'].isin([1,5])]
err_1_out = error_est_out(1, test_use, sol_transform0[0])
err_2_out = error_est_out(1, test_use, sol_transform1[0])
print('in sample error, lambda = 0.01 vs lambda = 1')
print(err_1, err_2)
print('out of sample error, lambda = 0.01 vs lambda = 1')
print(err_1_out, err_2_out)
