# import numpy
# import numpy.random as nrand
# vrand = list()
# vfirst = list()
# vmin = list()
# for i in range(0,100000):
# 	experiment = nrand.randint(2,size=[10,1000])
# 	count_head = numpy.sum(experiment,axis = 0)
# 	vrand.append(nrand.choice(count_head,1)[0])
# 	vfirst.append(count_head[0])
# 	vmin.append(min(count_head))

# print(numpy.mean(vmin), numpy.mean(vfirst), numpy.mean(vrand))
# print(min(vmin), max(vmin))
import math
sum = 0.0
for i in range(0,11):
	f = i/10
	sum += 1000*math.pow((1-f),999)*math.pow(f,2)
print(sum)