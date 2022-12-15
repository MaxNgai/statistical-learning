import numpy as np
import data
import util
from sklearn import linear_model
from sklearn.preprocessing import SplineTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


wage = data.wage()
boston = data.boston()


class NonLinear:


	
	def polyLinearRss(self, trx, trY, tsx, tsY):
		x = trx.reshape(-1,1)
		p = np.hstack([x,x**2,x**3,x**4])
		tsx = tsx.reshape(-1,1)
		testp = np.hstack([tsx,tsx**2,tsx**3,tsx**4])
		reg = linear_model.LinearRegression()

		reg.fit(p, trY)
		yhat = reg.predict(testp)
		rss = util.rss(tsY, yhat)
		return rss


	def polyLinearRssWithParam(self, trx, trY, tsx, tsY, param):
		x = trx.reshape(-1,1)
		tx = tsx.reshape(-1,1)
		p = np.empty([len(x),param])
	
		testp = np.empty([len(tx),param])
		for i in range(1, param+1):
			p[...,i-1] =  (x**i).reshape(-1)
			testp[...,i-1] =  (tx**i).reshape(-1)
		reg = linear_model.LinearRegression()
		reg.fit(p, trY)
		yhat = reg.predict(testp)
		rss = util.rss(tsY, yhat)
		return rss

	def polyLinear(self):
		rss = util.cv(wage.age, wage.Y, self.polyLinearRss)
		print(rss)


	
	def splineRssAge(self, trx, trY, tsx, tsY):
		spline = SplineTransformer(knots=np.asarray([ 0,25,40,60,100]).reshape(-1,1)) # a very different point is that here, knots input has to involve the boundary knot, which is 0 and 100
		splinedX = spline.fit_transform(trx.reshape(-1,1)) # and the spline is not like
		reg = linear_model.LinearRegression()
		reg.fit(splinedX, trY)
		yhat = reg.predict(spline.transform(tsx.reshape(-1,1)))
		rss = util.rss(tsY, yhat)
		return rss

	def splineRss(self, trx, trY, tsx, tsY, param):
		spline = SplineTransformer(degree = param) # a very different point is that here, knots input has to involve the boundary knot, which is 0 and 100
		splinedX = spline.fit_transform(trx.reshape(-1,1)) # and the spline is not like
		reg = linear_model.LinearRegression()
		reg.fit(splinedX, trY)
		yhat = reg.predict(spline.transform(tsx.reshape(-1,1)))
		rss = util.rss(tsY, yhat)
		return rss
	
	#p293
	def spline(self):
		rss = util.cv(wage.age, wage.Y, self.splineRssAge)
		print(rss)
		

	# gam but use bspline
	def gamRss(self, trx, trY, tsx, tsY):
		ageSpline = SplineTransformer(degree = 3)
		yearSpline = SplineTransformer(degree = 3)
		
		trainAge = ageSpline.fit_transform(trx[...,0].reshape(-1,1))
		trainYear = yearSpline.fit_transform(trx[...,1].reshape(-1,1))
		trainX = np.hstack([trainAge, trainYear, trx[...,2:]])
		reg = linear_model.LinearRegression()
		reg.fit(trainX, trY)

		testAge = ageSpline.transform(tsx[...,0].reshape(-1,1))
		testYear = yearSpline.transform(tsx[...,1].reshape(-1,1))
		testX = np.hstack([testAge, testYear, tsx[...,2:]])
		yhat = reg.predict(testX)
		rss = util.rss(tsY, yhat)
		return rss

	#p308
	def gam(self):
		
		xx = np.hstack([wage.age.reshape(-1,1), wage.year.reshape(-1,1)])
		e = wage.education.toarray()
		
		xx = np.hstack([xx, e])
		rss = util.cv(xx, wage.Y, self.gamRss)
		print(rss)

	#p288
	def polyLinearAll(self):
		x = wage.age.reshape(-1,1)
		p = np.hstack([x,x**2,x**3,x**4])
		
		
		reg = linear_model.LinearRegression()

		reg.fit(p, wage.Y)
		print(reg.coef_)
		print(reg.intercept_)


	#p299-6
	def polyLinearGetDegreeWithCv(self):
		for i in range(1, 6):
			print(util.cvWithParam(wage.age, wage.Y, self.polyLinearRssWithParam, i))

	#p299-7
	def boston(self):
		for i in range(1, 11):
			print(util.cvWithParam(boston.nox, boston.dis, self.polyLinearRssWithParam, i))

		print('--')

		for i in range(1, 7):
			print(util.cvWithParam(boston.nox, boston.dis, self.splineRss, i))







NonLinear().boston()
	