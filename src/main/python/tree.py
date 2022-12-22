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

carseat = data.carseat()
boston = data.boston()

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
		self.givenNLeaf(27)
		

	def givenNLeaf(self, nLeaf):
		train_x, test_x, train_y, test_y = train_test_split(carseat.X, carseat.high, test_size = 0.5, random_state=66)
		treee = DecisionTreeClassifier(random_state=0, max_leaf_nodes = nLeaf) # if doesnt set 27 will overfit
		treee.fit(train_x, train_y)
		yhat = treee.predict(test_x)
		errorRate = util.errorRate(test_y, yhat)
		print(errorRate)
		#util.confusionMatrix(test_y, yhat)

	#p327
	def growByNleaf(self):
		for i in range(2,65):
			self.givenNLeaf(i) #12, 13 leaf are the best

	def regressionTree0(self, nLeaf):
		r = DecisionTreeRegressor(random_state=0, max_leaf_nodes = nLeaf)
		train_x, test_x, train_y, test_y = train_test_split(boston.medvX, boston.medv, test_size = 0.5, random_state=66)
		r.fit(train_x, train_y)
		yhat = r.predict(test_x)
		mse = util.mse(test_y, yhat)
		print(mse)

	#p328
	def regressionTree(self):
		for i in range(2, 16):
			self.regressionTree0(i)

	def baggingRegressor0(self, mtry, nTree):
		r = BaggingRegressor(n_estimators=nTree, max_features = mtry, random_state=0)
		train_x, test_x, train_y, test_y = train_test_split(boston.medvX, boston.medv, test_size = 0.5, random_state=66)
		r.fit(train_x, train_y)
		yhat = r.predict(test_x)
		mse = util.mse(test_y, yhat)
		print(mse)
		return r

	#p329 bagging regressor
	def baggingRegressor(self):
		for i in range(1, 13):
			self.baggingRegressor0(i, 25) #i = 9 is the best 

		r9 = self.baggingRegressor0(9, 25)
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




treeBaseMethod().boosting()