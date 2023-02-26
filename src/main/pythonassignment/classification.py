from sklearn.linear_model import LogisticRegression as LR
import numpy as np
import data
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier

def confusionMatrix(y, yHat):

    tp=0
    tn=0
    fn=0
    fp=0
    for i in range(0, len(y)):
        if (yHat[i] == 1 and y[i] == 1):
            tp = tp + 1
        elif (yHat[i] == 1 and y[i] == 0):
            fp = fp + 1
        elif (yHat[i] == 0 and y[i] == 0):
            tn = tn + 1
        else:
            fn = fn + 1

    print([tp, fp, tn ,fn])


class Classification:
    defaultData = data.default()
    smarketData = data.smarket()
    caravan = data.caravan()
    weekly = data. weekly()
    auto = data. auto()

    # p134
    def logisticRegression(self):
        lr1 = LR()
        lr1.fit(self.defaultData.balance.reshape(-1,1), self.defaultData.default)
        print(lr1.coef_) # k
        print(lr1.intercept_) # b
        print(lr1.predict_proba(np.array(1000).reshape(-1,1))) # p(no), p(yes)
        print(lr1.classes_)

    # p135 和书本有点不一样，是求解方式的原因？
    def logisticRegressionStudent(self):
        lr = LR()
        lr.fit(self.defaultData.isStudent.reshape(-1,1), self.defaultData.default)
        print(lr.coef_) # k
        print(lr.intercept_) # b
        print(lr.predict_proba(np.array(1).reshape(-1,1))) # p(no), p(yes)
        print(lr.classes_)


    # p136 a little bit different from textbook, is it because how it solve the equation?
    def multivariateLR(self):
        lr = LR()
        t = self.defaultData.income / np.array([1000])
        input = np.hstack((t, self.defaultData.balance, self.defaultData.isStudent))
        lr.fit(input, self.defaultData.default)
        print(lr.coef_) # k
        print(lr.intercept_) # b
        print(lr.predict_proba(np.array([[40, 1500, 1]])))

    # p145
    def lda(self):
        lda = LinearDiscriminantAnalysis()
        lda.fit(self.defaultData.X, self.defaultData.default)
        yHat = lda.predict(self.defaultData.X)
        y = self.defaultData.default

        tp=0
        tn=0
        fn=0
        fp=0

        for i in range(0, len(y)):
            if (yHat[i] == "Yes" and y[i] == "Yes"):
                tp = tp + 1
            elif (yHat[i] == "Yes" and y[i] == "No"):
                fp = fp + 1
            elif (yHat[i] == "No" and y[i] == "No"):
                tn = tn + 1
            else:
                fn = fn + 1

        
        print([tp, tn, fp, fn])

    # p149
    def qda(self):
        qda = QuadraticDiscriminantAnalysis(store_covariance=True)
        qda.fit(self.defaultData.X, self.defaultData.default)
        yHat = qda.predict(self.defaultData.X)
        y = self.defaultData.default

        tp=0
        tn=0
        fn=0
        fp=0

        for i in range(0, len(y)):
            if (yHat[i] == "Yes" and y[i] == "Yes"):
                tp = tp + 1
            elif (yHat[i] == "Yes" and y[i] == "No"):
                fp = fp + 1
            elif (yHat[i] == "No" and y[i] == "No"):
                tn = tn + 1
            else:
                fn = fn + 1

        
        print([tp, tn, fp, fn])


    # p 156, different from textbook, but close enough
    def logisticRegressionOnSmarket(self):
        lr = LR(max_iter=100)
        lr.fit(self.smarketData.volumeAnd1to5, self.smarketData.direction)
        print(lr.coef_)
        print(lr.intercept_)

        y = self.smarketData.direction
        print(y)
        yHat = lr.predict(self.smarketData.volumeAnd1to5)

        tp=0
        tn=0
        fn=0
        fp=0
        for i in range(0, len(y)):
            if (yHat[i] == 1 and y[i] == 1):
                tp = tp + 1
            elif (yHat[i] == 1 and y[i] == 0):
                fp = fp + 1
            elif (yHat[i] == 0 and y[i] == 0):
                tn = tn + 1
            else:
                fn = fn + 1

        
        print([tp, tn, fp, fn])

    # p159
    def partlyTrainSmarketLR(self):
        lr = LR(max_iter=100)
        lr.fit(self.smarketData.volumeAnd1to5[0:998], self.smarketData.direction[0:998])
        


        y = self.smarketData.direction[998:1250]
        print(y)
        yHat = lr.predict(self.smarketData.volumeAnd1to5)

        confusionMatrix(y, yHat)

    # p 167, different from textbook
    def knn(self):
        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(self.caravan.trainX, self.caravan.trainY)
        yHat = neigh.predict(self.caravan.testX)
        y = self.caravan.testY

        confusionMatrix(y, yHat)
        

    #171-1
    def weeklyAssignment(self):
        lr = LR()
        lr.fit(self.weekly.trainX, self.weekly.trainY)
        yHat = lr.predict(self.weekly.testX)
        y = self.weekly.testY
        confusionMatrix(y, yHat) # 50% correct

        lda = LinearDiscriminantAnalysis()
        lda.fit(self.weekly.trainX, self.weekly.trainY)
        yHat = lda.predict(self.weekly.testX)
        confusionMatrix(y, yHat) # 50% correct

        qda = QuadraticDiscriminantAnalysis()
        qda.fit(self.weekly.trainX, self.weekly.trainY)
        yHat = qda.predict(self.weekly.testX)
        confusionMatrix(y, yHat) # 52% correct

        neigh = KNeighborsClassifier(n_neighbors=6)
        neigh.fit(self.weekly.trainX, self.weekly.trainY)
        yHat = neigh.predict(self.weekly.testX)
        confusionMatrix(y, yHat) # k=3, 45% correct
        #k=5,54%correct
        #k=1,49%correct
        #k=6 is the best, 57%

    def classificationOnAuto(self):
        y = self.auto.testY
        lda = LinearDiscriminantAnalysis()
        lda.fit(self.auto.trainX, self.auto.trainY)
        yHat = lda.predict(self.auto.testX)
        confusionMatrix(y, yHat)  # 88%

        qda = QuadraticDiscriminantAnalysis()
        qda.fit(self.auto.trainX, self.auto.trainY)
        yHat = qda.predict(self.auto.testX)
        confusionMatrix(y, yHat)  # 90%

        lr = LR()
        lr.fit(self.auto.trainX, self.auto.trainY)
        yHat = lr.predict(self.auto.testX)
        confusionMatrix(y, yHat) # 87%

        neigh = KNeighborsClassifier(n_neighbors=4)
        neigh.fit(self.weekly.trainX, self.weekly.trainY)
        yHat = neigh.predict(self.weekly.testX)
        confusionMatrix(y, yHat) #67%



Classification().classificationOnAuto()