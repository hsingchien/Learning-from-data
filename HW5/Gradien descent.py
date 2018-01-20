from math import exp
from math import pow
import matplotlib.pyplot as plt
# import numpy as np
i = 0
err = 1.0
u = 1.0
v = 1.0
errlist = [pow((u*exp(v)-2*v*exp(-u)),2)]
while(err > 1e-14 and i < 1000):
	gu = 2*(exp(v)+2*v*exp(-u))*(u*exp(v)-2*v*exp(-u))
	gv = 2*(u*exp(v)-2*v*exp(-u))*(u*exp(v)-2*exp(-u))
	u = u-0.1*gu 
	v = v-0.1*gv
	err = pow((u*exp(v)-2*v*exp(-u)),2)
	errlist.append(err)
	i = i+1

print(i)
print(err)
print(u,v)
x = list(range(0,11))
fig1 = plt.plot(x,errlist)


u2 = 1.0
v2 = 1.0
err_c = 1.0
errlist_c = [pow((u2*exp(v2)-2*v2*exp(-u2)),2)]
for j in range(0,30):
	gu_c = 2*(exp(v2)+2*v2*exp(-u2))*(u2*exp(v2)-2*v2*exp(-u2))
	u2 = u2 - 0.1 * gu_c
	err_c = pow((u2*exp(v2)-2*v2*exp(-u2)),2)
	errlist_c.append(err_c)
	gv_c = 2*(u2*exp(v2)-2*v2*exp(-u2))*(u2*exp(v2)-2*exp(-u2))
	v2 = v2 - 0.1 * gv_c
	err_c = pow((u2*exp(v2)-2*v2*exp(-u2)),2)
	errlist_c.append(err_c)


plt.figure(1)
plt.show(fig1)

plt.figure(2)
fig2 = plt.plot(list(range(0,61)), errlist_c)
plt.show(fig2)
print(err_c)





