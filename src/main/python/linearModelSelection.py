import numpy as np
import data
from sklearn import linear_model
from sklearn.linear_model import LassoCV
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.cross_decomposition import PLSRegression
from sklearn.cross_decomposition import PLSCanonical

hitters = data.hitters()

class LinearModelSelection:

	#p252, no the same as textbook
	def ridge(self):
		ridge = linear_model.Ridge(4)
		ridge.fit(hitters.X, hitters.Y)
		print(ridge.coef_)
		
		print(ridge.intercept_)

		print(np.sqrt(np.dot(ridge.coef_, ridge.coef_))) # l2-norm

		
	#p254, coefficient is not the same as the book but lambda is the same
	def ridgeCv(self):
		ridge = linear_model.RidgeCV(alphas=[0.1, 4, 50, 10**10]).fit(hitters.X, hitters.Y)
		'''this model first use cv to choose best lambda, then fit the entire data set to get coefficient'''
		print(ridge.coef_) 
		print(ridge.alpha_) # lambda with best score
		print(ridge.score(hitters.X.astype("float_"), hitters.Y.astype("float_"))) # r-square of model onto dataset, more like massive predict and evaluate
		print(ridge.best_score_) # negative mse, the greater the better


	#lasso,p255, not as textbook
	def lasso(self):
		lasso = LassoCV().fit(hitters.X, hitters.Y)
		print(lasso.alpha_) # best lambda
		print(lasso.coef_)
		print(lasso.score(hitters.X.astype("float_"), hitters.Y.astype("float_")))

	#p257
	def pcr(self):
		pcr = PCA(n_components = 7,svd_solver = 'full')
		print(pcr.fit_transform(hitters.X, hitters.Y)) #transform X to n*7 matrix
		print(pcr.components_) # direction vectors that has largest variation 
		
		mid = int(hitters.X.shape[0]/2)
		trainX = hitters.X[:mid, ...]
		testX = hitters.X[mid:, ...]
		trainY = hitters.Y[:mid]
		testY = hitters.Y[mid:]
		newX = pcr.fit_transform(trainX)
		reg = linear_model.LinearRegression()
		reg.fit(newX, trainY)

		dev = np.subtract(reg.predict(pcr.transform(testX)), (testY).astype('float_'))
		print(dev.dot(dev)/dev.shape[0])

	#p258 
	def pls(self):
		mid = int(hitters.X.shape[0]/2)
		trainX = hitters.X[:mid, ...]
		testX = hitters.X[mid:, ...]
		trainY = hitters.Y[:mid]
		testY = hitters.Y[mid:]

		pls = PLSRegression(n_components=3, scale = False)
		pls.fit(trainX, trainY)
		dev = np.subtract(pls.predict(testX), testY.astype('float_').reshape(-1,1))
		dev = dev.reshape(-1)
		print(np.dot(dev,dev)/ dev.shape)






		




		



LinearModelSelection().pls()