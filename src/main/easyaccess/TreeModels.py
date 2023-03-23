import numpy as np
from sklearn.ensemble import StackingClassifier
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier

#from sklearn.tree import DecisionTreeClassifier
#from sklearn.tree import DecisionTreeRegressor
def decisionTree(trainX, trainY, testX, testY):
    model = DecisionTreeClassifier(ccp_alpha=0)
    model.fit(trainX, trainY)
    yhat = model.predict(testX)
    indexOfLeaves = model.apply(testX)
    score = model.score(testX, testY)
    
    prob = model.predict_proba(testX)

    importance = model.feature_importances_
    leaves = model.get_n_leaves()
    depth = model.get_depth()

#from sklearn.ensemble import BaggingClassifier
#from sklearn.ensemble import BaggingRegressor
def baggingTrees(trainX, trainY, testX, testY):
    model = BaggingClassifier(n_estimators=100, max_features=1.0)
    model.fit(trainX, trainY)
    yhat = model.predict(testX)
    indexOfLeaves = model.apply(testX)
    score = model.score(testX, testY)    
    prob = model.predict_proba(testX)

    trees = model.estimators_

#from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import RandomForestClassifier
def randomForest(trainX, trainY, testX, testY):
    model = RandomForestClassifier(n_estimators=100, max_features='sqrt',ccp_alpha=0)
    yhat = model.predict(testX)
    indexOfLeaves = model.apply(testX)
    score = model.score(testX, testY)    
    prob = model.predict_proba(testX)

    trees = model.estimators_

#from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.ensemble import GradientBoostingRegressor
def gbdt(trainX, trainY, testX, testY):
    model = GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, subsample=1, max_depth=3)
    yhat = model.predict(testX)
    indexOfLeaves = model.apply(testX)
    score = model.score(testX, testY)    
    prob = model.predict_proba(testX)
    print(model.feature_importances_)
    trees = model.estimators_

#from sklearn.ensemble import StackingClassifier
#from sklearn.ensemble import StackingRegressor
def stackedGeneralization(trainX, trainY, testX, testY):
    estimators = [ GradientBoostingClassifier(), BaggingClassifier()]
    model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    yhat = model.predict(testX)
    score = model.score(testX, testY)    
    prob = model.predict_proba(testX)