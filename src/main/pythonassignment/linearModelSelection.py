import numpy as np
import data
from sklearn import linear_model
from sklearn.linear_model import LassoCV
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.cross_decomposition import PLSRegression
from sklearn.cross_decomposition import PLSCanonical
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge


hitters = data.hitters()

college = data.college()

boston = data.boston()

def mse(y, yhat):
	y = y.reshape(-1)
	yhat = yhat.reshape(-1)
	n = len(y)
	dev = np.subtract(y, yhat)
	return np.dot(dev, dev) / n

def rss(y, yhat):
	y = y.reshape(-1)
	yhat = yhat.reshape(-1)
	dev = np.subtract(y, yhat)
	return np.dot(dev, dev)

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


	#applied,p262-8
	def polynomialRegression(self):
		scaler = StandardScaler()

		x = stats.norm.rvs(0,1,size=100)
		e = stats.norm.rvs(0,1,size=100)
		x2 = np.power(x,2)
		x3 = np.power(x,3)
		Y = 4 + 3 * x + 2 * x2 + x3 + e
		X =np.asarray([])
		for i in range(10):
			X = np.hstack([X, np.power(x, i + 1)])
		X = (X.reshape(10, 100).T)
		scaler.fit(X)
		X = scaler.transform(X)



		lasso = LassoCV(max_iter=1000).fit(X, Y)
		print(lasso.alpha_) # best lambda
		print(lasso.coef_) # mostly are the first three  predictors

		y7 = 7 * np.power(x, 7) + e + 1

		lasso = LassoCV(max_iter=1000).fit(X, y7)
		print(lasso.alpha_) # best lambda
		print(lasso.coef_) # could not get the correct coef, though x^7 is included mostly


	#p263-9
	def collegeFit(self):
		reg = linear_model.LinearRegression()
		reg.fit(college.train_x, college.train_y)
		yhat = reg.predict(college.test_x)
		print(mse(college.test_y, yhat)) # ols regression mse

		ridge = linear_model.RidgeCV(alphas=[0.1, 1, 10, 100, 1000, 10000]).fit(college.X, college.Y)
		print(ridge.alpha_)
		newRidge = Ridge(alpha=ridge.alpha_) # get the best tunning param
		newRidge.fit(college.train_x, college.train_y) # re-train with train set
		ridgeHat = newRidge.predict(college.test_x)
		print(mse(college.test_y, ridgeHat)) # ridge regression mse

		lasso = LassoCV(max_iter=1000).fit(college.X, college.Y)
		print(lasso.alpha_) # best lambda
		print(lasso.coef_) # 10 predictors
		newLasso = linear_model.Lasso(alpha=lasso.alpha_) # get the best tunning param
		newLasso.fit(college.train_x, college.train_y) # re-train with train set
		lassoHat = newLasso.predict(college.test_x) 
		print(mse(college.test_y, lassoHat)) # lasso regression mse

		pcr = PCA(n_components = 'mle' , svd_solver = 'full')
		pcr.fit(college.train_x)
		pcr_test_x = pcr.transform(college.test_x)
		pcr_train_x = pcr.transform(college.train_x)
		print(pcr.n_components_) #
		reg.fit(pcr_train_x, college.train_y)
		pcrHat = reg.predict(pcr_test_x)
		print(mse(college.test_y, pcrHat)) # pcr regression mse, it is the worst

	
		pls = PLSRegression(n_components=13)
		pls.fit(college.train_x, college.train_y)
		plsHat = pls.predict(college.test_x)
		print(mse(college.test_y, plsHat)) # pls regression mse
			

		# best subset choose 8 predictors, [0, 1, 2, 3, 7, 8, 11, 15]. mse = 1162413
		# lasso is the best if exclude result from best subset


	# p264-11
	def boston(self):
		ridge = linear_model.RidgeCV(alphas=[ 0.1, 1, 10, 100, 1000, 10000]).fit(boston.X, boston.Y)
		print(ridge.alpha_)
		newRidge = Ridge(alpha=ridge.alpha_) # get the best tunning param
		newRidge.fit(boston.train_x, boston.train_y)
		ridgeHat = newRidge.predict(boston.test_x)
		print(mse(boston.test_y, ridgeHat)) # ridge regression mse

		lasso = LassoCV(max_iter=1000).fit(boston.X, boston.Y)
		print(lasso.alpha_) # best lambda
		print(lasso.coef_) # 4 predictors, [6,7,10,11]
		newLasso = linear_model.Lasso(alpha=lasso.alpha_) # get the best tunning param
		newLasso.fit(boston.train_x, boston.train_y)
		lassoHat = newLasso.predict(boston.test_x)
		print(mse(boston.test_y, lassoHat)) # ridge regression mse

		pcr = PCA(n_components = 'mle' , svd_solver = 'full')
		pcr.fit(boston.train_x)
		pcr_test_x = pcr.transform(boston.test_x)
		pcr_train_x = pcr.transform(boston.train_x)
		print(pcr.n_components_) #
		reg = linear_model.LinearRegression()
		reg.fit(pcr_train_x, boston.train_y)
		pcrHat = reg.predict(pcr_test_x)
		print(mse(boston.test_y, pcrHat)) # pcr regression mse

		#best subset is with 4 predictors,[0, 6, 7, 11]
		#so ridge is the best







		




		



LinearModelSelection().boston()