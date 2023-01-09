import matplotlib.pyplot as plt
import numpy as np
import util
import data
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay
from sklearn.linear_model import LogisticRegression as LR
from sklearn.preprocessing import PolynomialFeatures

auto = data.auto()
oj = data.oj()

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
		svc = SVC(kernel='linear', C = c) # the C is the cost param, the larger the  hyperplane grows thinner and more unstable
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


	def svmCv(self, x, y, kernel, c=1, test_size=0.5, degree=3, gamma='scale'):
		train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = test_size, random_state=66)
		svc = SVC(kernel=kernel, C = c, degree=degree, gamma=gamma) 
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


	#p369-4
	def polyVsRadial(self):
		np.random.seed(1)
		n = 100
		p = 2
		x = np.random.randn(n * 3).reshape(-1, 3)
		y = np.asarray(list(map(lambda e: 1 if e[0] ** 3 + e[2] * 1 > e[1] else 0, x))).reshape(-1,1) #x[..., 0], x[..., 1] are features
		xy = np.hstack([x[...,:2], y])
		x = x[..., :2]
		
		classOne = xy[xy[..., 2] == 0]
		classTwo = xy[xy[..., 2] == 1]
		
		util.scatter2dClasses((classOne[..., 0], classOne[..., 1]),(classTwo[..., 0], classTwo[..., 1]))

		self.svmCv(x, y, 'poly', degree=3)
		self.svmCv(x, y, 'rbf') # rbf is greater than poly

	#p369-5
	def kernelInLR(self):
		np.random.seed(1)
		n = 500
		p = 2
		x = np.random.rand(n * 3).reshape(-1, 3) - 0.5
		y = np.asarray(list(map(lambda e: 1 if e[0] ** 2 - e[1] ** 2 > 0 else 0, x))).reshape(-1,1)
		x = x[..., :2]
		xy = np.hstack([x[...,:2], y])
		classOne = xy[xy[..., 2] == 0]
		classTwo = xy[xy[..., 2] == 1]
		
		#util.scatter2dClasses((classOne[..., 0], classOne[..., 1]),(classTwo[..., 0], classTwo[..., 1]))
		self.lrCv(x, y) #0.536, worse than guess

		poly = PolynomialFeatures(3)
		newx = poly.fit_transform(x)
		self.lrCv(newx, y) #0.12, good

		self.svmCv(x, y, 'linear') # close to 0.5, like guess
		self.svmCv(x, y, 'poly', degree=2) # very good, 0.036
		self.svmCv(x, y, 'rbf')  # 0.064, also very good


	def lrCv(self, x, y, test_size = 0.5):
		train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = test_size, random_state=66)
		lr1 = LR()
		lr1.fit(train_x, train_y.reshape(-1))
		yhat = lr1.predict(test_x)
		error = util.errorRate(test_y, yhat)
		print(error)
	
	#p370-6
	def costParam(self):
		np.random.seed(1)
		n = 100
		p = 2
		x = np.random.randn(n * 3).reshape(-1, 3)
		y = np.asarray(list(map(lambda e: 1 if e[0] ** 2 * 0.25 + e[2] * 0.01 > e[1] else 0, x))).reshape(-1,1) #x[..., 0], x[..., 1] are features
		xy = np.hstack([x[...,:2], y])
		x = x[..., :2]
		
		classOne = xy[xy[..., 2] == 0]
		classTwo = xy[xy[..., 2] == 1]
		
		#util.scatter2dClasses((classOne[..., 0], classOne[..., 1]),(classTwo[..., 0], classTwo[..., 1]))

		c = [10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001] #c=0.1 is the best, which means a C-value larger than 1, i.e. stronger regularization can have better result.
		for i  in c:
			self.svmCv(x, y, 'linear', c=i)
		
	#p371-7
	def autoSeparate(self):
		x = auto.X
		y = auto.binaryY
		c = [10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001] #c=0.1 is the best, which means a C-value larger than 1, i.e. stronger regularization can have better result.
		degree = [1,2,3]
		gamma = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 2, 3,4]
		print("---Cost")
		for i  in c:
			self.svmCv(x, y, 'linear', c=i)
			pass

		print("---degree")
		for i in degree:
			self.svmCv(x, y, 'poly', c=i, degree=1)

		print("---gamma")
		for i in gamma:
			self.svmCv(x, y, 'rbf', gamma = i)

	#p372-8
	def tunningInOj(self):
		x = oj.X
		y = oj.Y
		self.svmCv(x, y, 'linear', c=0.01, test_size=270) #0.30 testError

		print('--linear')
		for i in range(20):
			self.svmCv(x, y, 'linear', c=float(i+1)/100, test_size=270) #c=0.04 is the best, errorRate is around 0.2
 
		print('--radial')
		for i in range(50):
			self.svmCv(x, y, 'rbf',  gamma=i+1/100,  test_size=270) # gamma=0.1 is the best, errorRate is around 0.28
			pass

		print('--poly')
		self.svmCv(x, y, 'poly', degree=3,  test_size=270)  # 


svm().tunningInOj()


