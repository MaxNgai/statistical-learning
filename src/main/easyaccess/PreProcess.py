from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import joblib  
from sklearn import linear_model
from sklearn.feature_selection import SequentialFeatureSelector



'''
Never invoke fit() method on test set!!!
绝不要对测试集调用fit方法

先切分了测试集和训练集，再预处理


'''


#from sklearn.preprocessing import StandardScaler
def standardization(x):
    scaler = StandardScaler()
    standard = scaler.fit_transform(x)


#from sklearn.preprocessing import OrdinalEncoder
def binaryEncode(x):
    enc = OrdinalEncoder()
    binary = enc.fit_transform(x)
    restoreX = enc.inverse_transform(binary)


#from sklearn.preprocessing import OneHotEncoder
def multiClassOneHot(x):
    enc = OneHotEncoder()
    multiClass = enc.fit_transform(x)
    restoreX = enc.inverse_transform(multiClass)

#from sklearn.decomposition import PCA
#from sklearn.preprocessing import StandardScaler
def PCA(x):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(x)
    
    pca = PCA()
    trans = pca.fit_transform(scaled)
    print(pca.components_) # n_component * n_feature
    print(trans[0]) # Z, pca score
    print(pca.explained_variance_ratio_)


#from sklearn.model_selection import train_test_split
def trainTestSplit(x,y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)


#from sklearn.model_selection import cross_val_score
def quickEstimateModel(model, x, y, cv=10):
    scores = cross_val_score(model, x, y, cv=cv)
    scores.mean()
    #return an array with size of cv, elements as testset score

#from sklearn.model_selection import KFold
def kfold(x,y,k):
    kf = KFold(n_splits=k)
    for train, test in kf.split(x,y):
        pass

    
# import joblib
def persist(fileName):
    d = np.asarray([1,2,2,4]).reshape(-1,2)
    model = linear_model.LinearRegression()
    model.fit(d[...,0].reshape(-1,1), d[...,1].reshape(-1,1))

    s = joblib.dump(model, fileName + ".model")

# import joblib
def readFromPersist(fileName):
    model = joblib.load(fileName  + ".model")
    yhat = model.predict(np.asarray([3]).reshape(-1,1))
    print(yhat)

#from sklearn.feature_selection import SequentialFeatureSelector
def sequentialFeatureSelection(model,x,y):
    selector = SequentialFeatureSelector(model, n_features_to_select = 3, direction="forward")
    selector.fit(x,y)
    selectedIndex = selector.get_support()


#from sklearn.model_selection import GridSearchCV
#from sklearn.linear_model import LogisticRegression
#scoring: https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules
def tuningHyperParameter():
    iris = load_iris()
    
    model = LogisticRegression() #10folds; 10 picks of inverse of λ; penalty type
    
    parameters = {
    "C":[1,10,0.1,100,1000],
    "penalty":['l1','l2']
    }
    cvModel = GridSearchCV(model, param_grid = parameters, scoring='r2') # scoring can be 'neg_mean_squared_error', 'r2', 'roc_auc','precision', 'recall'
    cvModel.fit(iris.data, iris.target)
    #print(cvModel.best_params_)
    print(cvModel.best_index_)
    #print(cvModel.cv_results_['params'])
    print(cvModel.cv_results_['mean_test_score'])


tuningHyperParameter()