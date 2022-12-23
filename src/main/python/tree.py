from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import linear_model
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
import matplotlib.pyplot as plt
import numpy as np
import data
import util
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
import graphviz 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingRegressor
from sklearn.inspection import permutation_importance
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import plot_tree

carseat = data.carseat()
boston = data.boston()
oj = data.oj()
hitters = data.hitters()
caravan = data.caravan()

class treeBaseMethod:

	#p325
	def classificationTree(self):
		treee = DecisionTreeClassifier(random_state=0, max_leaf_nodes = 27) # if doesnt set 27 will overfit
		treee.fit(carseat.X, carseat.high)
		print(treee.get_n_leaves()) # 61
		print(export_text(treee))
		yhat = treee.predict(carseat.X)
		errorRate = util.errorRate(carseat.high, yhat)
		print(errorRate)

	#p326
	def classificationTreeCv(self):
		self.classificationTree0(27, carseat.X, carseat.high)
		

	def classificationTree0(self, nLeaf, x, y, test_ratio = 0.5):
		train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = test_ratio, random_state=66)
		treee = DecisionTreeClassifier(random_state=0, max_leaf_nodes = nLeaf) # if doesnt set 27 will overfit
		treee.fit(train_x, train_y)
		yhat = treee.predict(test_x)
		errorRate = util.errorRate(test_y, yhat)
		print(str(errorRate) + " nLeaf = " + str(nLeaf))
		#util.confusionMatrix(test_y, yhat)
		return treee,errorRate

	#p327
	def growByNleaf(self):
		for i in range(2,65):
			self.classificationTree0(i, carseat.X, carseat.high) #12, 13 leaf are the best

	def regressionTree0(self, nLeaf, x, y):
		r = DecisionTreeRegressor(random_state=0, max_leaf_nodes = nLeaf)
		train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.5, random_state=66)
		r.fit(train_x, train_y)
		yhat = r.predict(test_x)
		mse = util.mse(test_y, yhat)
		print(str(mse) + " nLeaf = " + str(nLeaf))

	#p328
	def regressionTree(self):
		for i in range(2, 16):
			self.regressionTree0(i, boston.medvX, boston.medv)

	def baggingRegressor0(self, mtry, nTree, x, y):
		r = BaggingRegressor(n_estimators=nTree, max_features = mtry, random_state=0)
		train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.5, random_state=66)
		r.fit(train_x, train_y)
		yhat = r.predict(test_x)
		mse = util.mse(test_y, yhat)
		print(str(mse) + " features = " + str(mtry))
		return r

	#p329 bagging regressor
	def baggingRegressor(self):
		for i in range(1, 13):
			self.baggingRegressor0(i, 25, boston.medvX, boston.medv) #i = 9 is the best 

		r9 = self.baggingRegressor0(9, 25, boston.medvX, boston.medv)
		train_x, test_x, train_y, test_y = train_test_split(boston.medvX, boston.medv, test_size = 0.5, random_state=66)

		res = permutation_importance(r9, test_x,test_y, n_repeats=10, random_state=0)
		print(res.importances_mean)


	def boosting(self):
		r = AdaBoostRegressor(random_state=0, n_estimators=50, learning_rate=1)
		train_x, test_x, train_y, test_y = train_test_split(boston.medvX, boston.medv, test_size = 0.5, random_state=66)
		r.fit(train_x, train_y)
		yhat = r.predict(test_x)
		mse = util.mse(test_y, yhat)
		print(mse)




	#p333-8
	def regressionTreeOnCarseat(self):
		for i in range(2, 50):
			self.regressionTree0(i, carseat.X, carseat.Y) # round nLeaf = 20 is that best

		print("bagging ↓")
		for i in range(2, 13):
			r = self.baggingRegressor0(i, 100, carseat.X, carseat.Y)

		r9 = self.baggingRegressor0(12, 100, carseat.X, carseat.Y)
		train_x, test_x, train_y, test_y = train_test_split(carseat.X, carseat.Y, test_size = 0.5, random_state=66)

		res = permutation_importance(r9, test_x,test_y, n_repeats=10, random_state=55)
		print(res.importances_mean) # ShelveLoc & price matters most

	#p334-9
	def classficationOnOJ(self):
		mse = np.empty(28)
		regressor = list()
		for i in range(2,30):
			treee, errorRate = self.classificationTree0(i, oj.X, oj.Y, 0.2) #leaf=10 or 11 is the best
			mse[i - 2] = errorRate
			regressor.append(treee)

		util.plot(mse)		
		util.plotTree(regressor[8])
		

	def boosting0(self, x, y, nTree, learning_rate, test_size):
		r = AdaBoostRegressor(random_state=0, n_estimators=nTree, learning_rate=learning_rate)
		train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = test_size, random_state=66)
		r.fit(train_x, train_y)
		yhat = r.predict(test_x)
		mse = util.mse(test_y, yhat)
		print(mse)
		return mse, r

	def ridgeOnHitters(self):
		ridge = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1,10,100,1000]).fit(hitters.rawX, hitters.logY)
		print(ridge.alpha_)
		print(ridge.predict(hitters.rawX))

	def lassoOnHitters(self):
		lasso = LassoCV(alphas=[1e-3, 1e-2, 1e-1, 1,10,100,1000]).fit(hitters.rawX, hitters.logY)
		print(lasso.alpha_)
		print(lasso.predict(hitters.rawX))

	def ridge0(self, x, y, alpha, test_size=0.5):
		r = Ridge(alpha=alpha) 
		train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = test_size, random_state=66)
		r.fit(train_x, train_y)
		yhat = r.predict(test_x)
		mse = util.mse(test_y, yhat)
		print(mse)
		return mse

	def lasso0(self, x, y, alpha, test_size=0.5):
		r = linear_model.Lasso(alpha = alpha)
		train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = test_size, random_state=66)
		r.fit(train_x, train_y)
		yhat = r.predict(test_x)
		mse = util.mse(test_y, yhat)
		print(mse)
		return mse


	def ols0(self, x, y,  test_size=0.5):
		train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = test_size, random_state=66)
		reg = linear_model.LinearRegression()
		reg.fit(train_x, train_y)
		yhat = reg.predict(test_x)
		mse = util.mse(test_y, yhat)
		print(mse)
		return mse




	#p335-10, 12
	def boostingOnHitters(self):
		lRange = 200
		m = np.empty(lRange)
		'''
		for lam in range(1, lRange+1):
			mse, r = self.boosting0(hitters.rawX, hitters.logY, 100, lam/1000, 0.33) # lambda = 0.1 is the best
			m[lam - 1] = mse


		util.plot(m)
'''
		boostingMse, boostingRegressor = self.boosting0(hitters.rawX, hitters.logY, 100, 0.1, 0.33)
		print("--")
		print(boostingMse)
		res = permutation_importance(boostingRegressor,hitters.rawX, hitters.logY, n_repeats=10, random_state=0)
		print(res.importances_mean)

		# ols is the worst
		self.ols0(hitters.rawX, hitters.logY) 

		# ridge is only slightly better than ols
		self.ridge0(hitters.rawX, hitters.logY, 10) # lambda = 10 is got from cv

		# lasso is a bit worse than ridge but better than ols
		self.lasso0(hitters.rawX, hitters.logY, 0.01) # lambda = 0.01 is got from cv

		
		# bagging is much better than ols, but a bit worse than boosting
		self.baggingRegressor0(11, 100, hitters.rawX, hitters.logY) # 11 is got from cv
		

	def boostingClassify0(self,x,y,nTree,learning_rate, test_size):
		train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = test_size, random_state=66)
		c = AdaBoostClassifier(n_estimators = nTree, learning_rate = learning_rate)
		c.fit(train_x, train_y)
		yhat = c.predict(test_x)
		mse = util.errorRate(test_y, yhat)
		print(mse)
		
		util.confusionMatrixNumeric(test_y, yhat)
		return mse, c

	def knn0(self,x,y, k, test_size):
		train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = test_size, random_state=66)
		neigh = KNeighborsClassifier(n_neighbors=k)
		neigh.fit(train_x, train_y)
		yhat = neigh.predict(test_x)
		mse = util.errorRate(test_y, yhat)
		print(mse)
		util.confusionMatrixNumeric(test_y, yhat)
		return mse




	def boostingOnCaravan(self):
		mse, c = self.boostingClassify0(caravan.rawX, caravan.rawY, 1000, 0.01, 0.83) # all are false
		#importance = permutation_importance(c, caravan.rawX, caravan.rawY, n_repeats=3, random_state=0)
		#print(importance.importances_mean())

		m = np.empty(99)
		for k in range(1, 10):
			m[k-1] = self.knn0(caravan.rawX, caravan.rawY, k , 0.83)
		util.plot(m) 

		#sparse data causing??


			







treeBaseMethod().boostingOnCaravan()