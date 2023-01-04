import matplotlib.pyplot as plt
import numpy as np
import util
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay


class svm:
	#p359
	def trySvc(self):
		np.random.seed(1)
		x = np.random.randn(40).reshape(-1, 2)
		y = np.vstack([np.ones(10) * -1, np.ones(10)]).reshape(-1,1)

		for i in range(10,20):
			x[i, 0] = x[i, 0] + 1
			x[i, 1] = x[i, 1] + 1

		#util.scatter2dClasses((x[0:10, 0], x[0:10, 1]), (x[10:20, 0], x[10:20, 1]))

		svc = SVC(kernel='linear', C = 1) # the C is the cost param, the larger the  hyperplane grows thiner
		svc.fit(x,y)
		yhat = svc.predict(x)
		error = util.errorRate(y, yhat)
		print(error) # only 1 is incorrect

		print(svc.support_) # indice of support vector

	def trySvcButLessSeparated(self):
		np.random.seed(1)
		x = np.random.randn(40).reshape(-1, 2)
		y = np.vstack([np.ones(10) * -1, np.ones(10)]).reshape(-1,1)

		for i in range(10,20):
			x[i, 0] = x[i, 0] + 0.5
			x[i, 1] = x[i, 1] + 0.5

		#util.scatter2dClasses((x[0:10, 0], x[0:10, 1]), (x[10:20, 0], x[10:20, 1]))

		cs = np.asarray([0.0001, 0.001, 0.1, 1, 10, 100, 1000])
		for i in cs:
			self.svcCv(x,y,i)

	def svcCv(self, x, y, c, test_size=0.5):
		train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = test_size, random_state=66)
		svc = SVC(kernel='linear', C = c) # the C is the cost param, the larger the  hyperplane grows thiner
		svc.fit(train_x, train_y.reshape(-1))
		yhat = svc.predict(test_x)
		error = util.errorRate(test_y, yhat)
		print(error) #C = 1 is the best,

	#p363
	def svm(self):
		np.random.seed(1)
		x = np.random.randn(400).reshape(-1, 2)
		y = np.hstack([np.ones(150), np.ones(50) * 2]).reshape(-1,1)
		x[0:100] = x[0:100] + 2
		x[100:150] = x[100:150] - 2
		#util.scatter2dClasses((x[0:150,0], x[0:150,1]), (x[150:,0], x[150:,1]))

		cs = np.asarray([0.0001, 0.001, 0.1, 1, 10, 100, 1000])
		for i in cs:
			self.svmCv(x,y,'rbf', i) #errorRate = 0.08
			self.svmCv(x,y,'poly', i ) # errorRate = 0.25


	def svmCv(self, x, y, kernel, c=1, test_size=0.5):
		train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = test_size, random_state=66)
		svc = SVC(kernel=kernel, C = c) 
		svc.fit(train_x, train_y.reshape(-1))
		yhat = svc.predict(test_x)
		error = util.errorRate(test_y, yhat)
		print(error) 
	
	#p365
	def roc(self):
		np.random.seed(1)
		x = np.random.randn(400).reshape(-1, 2)
		y = np.hstack([np.ones(150), np.ones(50) * 2]).reshape(-1,1)
		x[0:100] = x[0:100] + 2
		x[100:150] = x[100:150] - 2

		train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.5, random_state=66)
		svc = SVC(kernel='rbf', C = 1) 
		svc.fit(train_x, train_y.reshape(-1))
		yhat = svc.predict(test_x)
		error = util.errorRate(test_y, yhat)
		print(error) 

		RocCurveDisplay.from_estimator(svc, test_x, test_y)
		plt.show()

	#p366
	def multiClass(self):
		np.random.seed(1)
		x = np.random.randn(400).reshape(-1, 2)
		y = np.hstack([np.ones(150), np.ones(50) * 2]).reshape(-1,1)
		x[0:100] = x[0:100] + 2
		x[100:150] = x[100:150] - 2

		x = np.vstack([x, np.random.randn(100).reshape(-1, 2)])
		y = np.vstack([y, np.zeros(50).reshape(-1,1)])
		x[200:, 1] = x[200:, 1] + 2

		train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.5, random_state=66)
		svc = SVC(kernel='rbf', C = 1, gamma=0.1) 
		svc.fit(train_x, train_y.reshape(-1))
		yhat = svc.predict(test_x)
		error = util.errorRate(test_y, yhat)
		print(error) 


	



svm().multiClass()


