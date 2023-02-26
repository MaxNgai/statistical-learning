import numpy as np
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#from sklearn import linear_model
def linearFit(trainX, trainY, testX, testY):
    model = linear_model.LinearRegression()
    model.fit(trainX, trainY)
    yhat = model.predict(testX)
    rsquare = model.score(testX, testY)

#from sklearn.linear_model import Ridge
def ridgeFit(trainX, trainY, testX, testY):
    model = linear_model.Ridge(alpha=1)
    model.fit(trainX, trainY)
    yhat = model.predict(testX)
    rsquare = model.score(testX, testY)
    coef = model.coef_

#from sklearn.linear_model import RidgeCV
def ridgeCVFit(trainX, trainY, testX, testY):
    model = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])
    model.fit(trainX, trainY)
    yhat = model.predict(testX)
    rsquare = model.score(testX, testY)
    coef = model.coef_      
    bestAlpha = model.alpha_        


#from sklearn import linear_model
def lassoFit(trainX, trainY, testX, testY):
    model = linear_model.Lasso(alpha=0.1)
    model.fit(trainX, trainY)
    yhat = model.predict(testX)
    rsquare = model.score(testX, testY)
    coef = model.coef_

#from sklearn.linear_model import LassoCV
def lassoCVFit(trainX, trainY, testX, testY):
    model = LassoCV(cv=10, eps=1e-3,n_jobs=4) #cv 10 folds, automated search best alpha, parallelism=4
    model.fit(trainX, trainY)
    yhat = model.predict(testX)
    rsquare = model.score(testX, testY)
    coef = model.coef_

#from sklearn.linear_model import LogisticRegression
def logisticsRegression(trainX, trainY, testX, testY):
    model = LogisticRegression(penalty='l2', C=1) # born with l2-pernalty, C is inverse of λ
    model.fit(trainX, trainY)
    yhat = model.predict(testX)
    rsquare = model.score(testX, testY)
    coef = model.coef_
    prob = model.predict_proba(testX)

#from sklearn.linear_model import LogisticRegressionCV
def LogisticRegressionCv(trainX, trainY, testX, testY):
    model = LogisticRegressionCV(cv=10, Cs=100, penalty='l2') #10folds; 10 picks of inverse of λ; penalty type
    model.fit(trainX, trainY)
    yhat = model.predict(testX)
    rsquare = model.score(testX, testY)
    coef = model.coef_
    prob = model.predict_proba(testX)

#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
def LDA(trainX, trainY, testX, testY):
    model = LinearDiscriminantAnalysis()
    model.fit(trainX, trainY)
    yhat = model.predict(testX)
    rsquare = model.score(testX, testY)
    coef = model.coef_
    prob = model.predict_proba(testX)

#from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
def QDA(trainX, trainY, testX, testY):
    model = QuadraticDiscriminantAnalysis()
    model.fit(trainX, trainY)
    yhat = model.predict(testX)
    rsquare = model.score(testX, testY)
    coef = model.coef_
    prob = model.predict_proba(testX)