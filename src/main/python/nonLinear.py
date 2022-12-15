import numpy as np
import data
import util
from sklearn import linear_model
from sklearn.preprocessing import SplineTransformer
from sklearn.model_selection import train_test_split

wage = data.wage()


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

	def polyLinear(self):
		rss = util.cv(wage.age, wage.Y, self.polyLinearRss)
		print(rss)


	
	def splineRss(self, trx, trY, tsx, tsY):
		spline = SplineTransformer(knots=np.asarray([ 0,25,40,60,100]).reshape(-1,1)) # a very different point is that here, knots input has to involve the boundary knot, which is 0 and 100
		splinedX = spline.fit_transform(trx.reshape(-1,1)) # and the spline is not like
		reg = linear_model.LinearRegression()
		reg.fit(splinedX, trY)
		yhat = reg.predict(spline.transform(tsx.reshape(-1,1)))
		rss = util.rss(tsY, yhat)
		return rss
	
	#p293
	def spline(self):
		rss = util.cv(wage.age, wage.Y, self.splineRss)
		print(rss)
		

	# gam but use bspline
	def gamRss(self, trx, trY, tsx, tsY):
		ageSpline = SplineTransformer(degree = 3)
		yearSpline = SplineTransformer(degree = 3)
		trainAge = ageSpline.fit_transform(trx[...,0].reshape(-1,1))
		trainYear = yearSpline.fit_transform(trx[...,1].reshape(-1,1))
		trainX = np.hstack([trainAge, trainYear])
		reg = linear_model.LinearRegression()
		reg.fit(trainX, trY)

		testAge = ageSpline.transform(tsx[...,0].reshape(-1,1))
		testYear = yearSpline.transform(tsx[...,1].reshape(-1,1))
		testX = np.hstack([testAge, testYear])
		yhat = reg.predict(testX)
		rss = util.rss(tsY, yhat)
		return rss

	def gam(self):
		rss = util.cv(np.hstack([wage.age.reshape(-1,1), wage.year.reshape(-1,1)]), wage.Y, self.gamRss)
		print(rss)


NonLinear().polyLinear()
NonLinear().spline()
NonLinear().gam()


