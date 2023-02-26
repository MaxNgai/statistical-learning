import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.preprocessing import SplineTransformer


#from sklearn.preprocessing import PolynomialFeatures
#from sklearn import linear_model
def polynomialRegression(x,y):
    poly = PolynomialFeatures(3)
    polyX = poly.fit_transform(x)
    model = linear_model.LinearRegression()
    model.fit(polyX, y)
    yhat = model.predict(polyX)
    rsquare = model.score(polyX, y)

#from sklearn.preprocessing import SplineTransformer
def splineRegression(x,y):
    spline = SplineTransformer(degree=3, n_knots=3)
    splineX = spline.fit_transform(x)
    model = linear_model.LinearRegression()
    model.fit(splineX, y)
    yhat = model.predict(splineX)
    rsquare = model.score(splineX, y)