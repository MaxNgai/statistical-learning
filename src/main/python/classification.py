from sklearn.linear_model import LogisticRegression as LR
import numpy as np
import data
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class Classification:
	defaultData = data.default()

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


	# p136 和书本有点不一样，是求解方式的原因？
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



Classification().lda()