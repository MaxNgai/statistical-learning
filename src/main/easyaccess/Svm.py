import numpy as np
from sklearn.svm import SVC



#from sklearn.svm import SVC
def svmClassificationt(trainX, trainY, testX, testY):
    # kernel in ('rbf','poly','lnear')
    # C is the inverse of regularization strength, the greater the more unstable
    # degree is only for poly
    model = SVC(kernel='rbf', C = c, degree=3) 
    model.fit(trainX, trainY)
    yhat = model.predict(testX)
    score = model.score(testX, testY)
    prob = model.predict_proba(testX)

