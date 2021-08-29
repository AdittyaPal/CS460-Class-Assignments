import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import shutil

def StochasticGD(x, y, noOfIterations=20, alpha=10e-3, b=10, w=0):
	noOfSamples=len(x)
	frames=[]
	k=0
	for i in range(noOfIterations):
		shuffle=np.random.permutation(noOfSamples)
		for j in range(noOfSamples):
			diff_w=-2*x[shuffle[j]]*(y[shuffle[j]]-(b+w*x[shuffle[j]]))
			diff_b=-2*(y[shuffle[j]]-(b+w*x[shuffle[j]]))
			w=w-alpha*diff_w
			b=b-alpha*diff_b

			temp_x=np.array([0,10])
			temp_y=b+w*temp_x
			fig, ax=plt.subplots(figsize=(10,8))
			plt.scatter(x, y)
			plt.scatter(x[shuffle[j]], y[shuffle[j]], c='red')
			plt.plot(temp_x, temp_y, label='By Stochastic Gradient Descent')
			plt.text(0.1, 18, "Iteration: {:03d}".format(k), size=18)
			plt.text(0.1, 17, "$w: {:.3f}$".format(w), size=18)
			plt.text(0.1, 16, "$b: {:.3f}$".format(b), size=18)
			plt.xlabel('x')
			plt.ylabel('y')
			plt.ylim([0, 20])
			plt.grid(True)
	
			frame="tmp/{:30d}.png".format(k)
			frames.append(frame)
			fig.savefig(frame, dpi=40)
			plt.close()	
			k+=1		
	return frames

noOfSamples=100
mean=np.array([5.0, 10.0])
corr=np.array([[  3.40, -2.75],	[ -2.75,  5.500]])
x, y=np.random.multivariate_normal(mean, corr, size=noOfSamples).T

if os.path.exists('tmp'):
	shutil.rmtree('tmp')
	
os.mkdir('tmp')

frames=StochasticGD(x, y)

# creating animation
with imageio.get_writer("Stochastic.gif", mode="I", fps=5) as writer:
	for name in frames:
		im = imageio.imread(name)
		writer.append_data(im)
# cleaning up
for item in frames:
	if item.endswith(".png"):
		os.remove(item)
print(".gif successfully created!")

