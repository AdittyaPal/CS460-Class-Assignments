import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

def BatchGD(x, y, noOfIterations=2000, alpha=10e-3, b=15, w=-1):
	cost=np.zeros(noOfIterations)
	for i in range(noOfIterations):
		cost[i]=np.sum(np.square(y-(b+w*x)))/(noOfSamples)
		diff_w=np.sum(-2*x*(y-(b+w*x)))/noOfSamples
		diff_b=np.sum(-2*(y-(b+w*x)))/noOfSamples
		w=w-alpha*diff_w
		b=b-alpha*diff_b
	return [w, b], cost

def StochasticGD(x, y, noOfIterations=2000, alpha=10e-3, b=15, w=-1):
	cost=np.zeros(noOfIterations)
	noOfSamples=len(x)
	for i in range(noOfIterations):
		shuffle=np.random.permutation(noOfSamples)
		cost[i]=np.sum(np.square(y-(b+w*x)))/(noOfSamples)
		for j in range(noOfSamples):
			diff_w=-2*x[shuffle[j]]*(y[shuffle[j]]-(b+w*x[shuffle[j]]))
			diff_b=-2*(y[shuffle[j]]-(b+w*x[shuffle[j]]))
			w=w-alpha*diff_w
			b=b-alpha*diff_b
	return [w, b], cost

def getMiniBatches(x, y, batchSize, noOfSamples, randomize=True):
	if randomize==True:
		shuffle=np.random.permutation(noOfSamples)
	for start in range(0, noOfSamples, batchSize):
		end=min(start+batchSize, noOfSamples)
		if randomize==True:
			batch=shuffle[start:end]
		else:
			batch=slice(start, end)
		yield x[batch], y[batch]

def miniBatchGD(x, y, batchSize=20, noOfIterations=2000, alpha=10e-3, b=15, w=-1):
	cost=np.zeros(noOfIterations)
	noOfSamples=len(x)
	noBatches=noOfSamples//batchSize
	
	for i in range(noOfIterations):
		cost[i]=np.sum(np.square(y-(b+w*x)))/(noOfSamples)
		for x_batch, y_batch in getMiniBatches(x, y, batchSize, noOfSamples):
			diff_w=np.sum(-2*x_batch*(y_batch-(b+w*x_batch)))/batchSize
			diff_b=np.sum(-2*(y_batch-(b+w*x_batch)))/batchSize
			w=w-alpha*diff_w
			b=b-alpha*diff_b
	return [w, b], cost

noOfSamples=100
mean=np.array([5.0, 10.0])
corr=np.array([[  3.40, -2.75],	[ -2.75,  5.500]])
x, y=np.random.multivariate_normal(mean, corr, size=noOfSamples).T

#create a linear regression object
regress=linear_model.LinearRegression()

regress.fit(x.reshape(-1,1), y)
y_predict=regress.predict(x.reshape(-1,1))

theta_BatchGD, cost_BatchGD=BatchGD(x, y)
y_predictByBatchGD=theta_BatchGD[0]*x+theta_BatchGD[1]

theta_StochasticGD, cost_StochasticGD=StochasticGD(x, y, alpha=10e-4)
y_predictByStochasticGD=theta_StochasticGD[0]*x+theta_StochasticGD[1]

theta_miniBatchGD, cost_miniBatchGD=miniBatchGD(x, y)
y_predictByMiniBatchGD=theta_miniBatchGD[0]*x+theta_miniBatchGD[1]

print('By scikit-learn:')
print(regress.coef_)
print(regress.intercept_)
print('By Batch Gradient Descent:')
print(theta_BatchGD[0])
print(theta_BatchGD[1])
print('By Stochastic Gradient Descent:')
print(theta_StochasticGD[0])
print(theta_StochasticGD[1])
print('By Mini Batch Gradient Descent:')
print(theta_miniBatchGD[0])
print(theta_miniBatchGD[1])

plt.figure(figsize=(8,10))
plt.subplot(2,1,1)
plt.plot(x, y, '.')
plt.plot(x, y_predict, label='By scikit-learn', linewidth=0.5)
plt.plot(x, y_predictByBatchGD, label='By Batch Gradient Descent', linewidth=0.5)
plt.plot(x, y_predictByStochasticGD, label='By Stochastic Gradient Descent', linewidth=0.5)
plt.plot(x, y_predictByMiniBatchGD, label='By Mini Batch Gradient Descent', linewidth=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()

plt.subplot(2,1,2)
plt.plot(np.arange(len(cost_BatchGD)), cost_BatchGD, label='Batch Gradient Descent', linewidth=0.5)
plt.plot(np.arange(len(cost_StochasticGD)), cost_StochasticGD, label='Stochastic Gradient Descent', linewidth=0.5)
plt.plot(np.arange(len(cost_miniBatchGD)), cost_miniBatchGD, label='Mini Batch Gradient Descent', linewidth=0.5)
plt.xlabel('No. of Iteraions')
plt.ylabel('Cost')
plt.legend()

plt.show()