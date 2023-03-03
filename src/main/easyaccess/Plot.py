import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import RocCurveDisplay
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split




def style():
    plt.style.use('_mpl-gallery') #grid style

# dot to line
def dotToCurve(x, y):
    fig, ax = plt.subplots()
    ax.plot(x,y)
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    ax.legend()
    plt.show()

# scatter
def scatter(x, y):
    fig, ax = plt.subplots()
    ax.scatter(x,y)
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    ax.legend()
    plt.show()

def scatter3D(x,y,z):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter(x,y,z)
    ax.legend()
    plt.show()

def bar(x,y):
    fig, ax = plt.subplots()
    ax.bar(x,y)
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    ax.legend()
    plt.show()

def hist(x):
    fig, ax = plt.subplots()
    ax.hist(x, edgecolor="white")
    plt.xlabel('X')
    plt.ylabel('frequency')
    ax.legend()
    plt.show()

def boxplot(x):
    # x is an array of array
    fig, ax = plt.subplots()
    ax.boxplot(x)
  
    ax.legend()
    plt.show()


def residualScatter(y, ybar):
    residual = np.abs(y-ybar)

    fig, ax = plt.subplots()
    ax.scatter(ybar,residual)
    plt.xlabel('yBar')
    plt.ylabel('Residual')
    plt.show()


#from sklearn.metrics import RocCurveDisplay
def roc(model, testX, testY):
    RocCurveDisplay.from_estimator(model, testX, testY)
    plt.show()


def homemadeROC():
    X, y = load_wine(return_X_y=True)
    y = y==2


    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    svc = SVC(random_state=42, probability=True)
    svc.fit(X_train, y_train)

    prob = svc.predict_proba(X_test)


    threshold = np.asarray(range(101)) / 100
    res = []
    for t in threshold:
        yhat = prob[...,0] < t
        tp = 0
        tn = 0
        total = len(yhat)
        for index, item in enumerate(yhat):
            if (item and y_test[index]):
                tp = tp + 1
            elif (item == False and y_test[index] == False):
                tn = tn + 1
        res.append(tp/total)
        res.append(1 - tn/total)

    co = np.asarray(res).reshape(-1, 2)
    print(co)
    plt.plot(co[...,1] ,co[...,0])
    plt.show()
