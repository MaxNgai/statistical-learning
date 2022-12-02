import numpy as np
import data
from sklearn import linear_model
from sklearn.linear_model import LassoCV

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




		



LinearModelSelection().lasso()