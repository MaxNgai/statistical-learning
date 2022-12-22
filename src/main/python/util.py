from sklearn.model_selection import train_test_split
import numpy as np


def cv(x, y, rssGetter):
	k = 10
	rss = np.empty([k])
	for i in range(k):
		x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/k, shuffle = False)
		rss0 = rssGetter(x_train, y_train, x_test, y_test)
		rss[i] = rss0
	return np.mean(rss)

def cvWithParam(x, y, rssGetter, p):
	k = 10
	rss = np.empty([k])
	for i in range(k):
		x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/k, shuffle = False)
		rss0 = rssGetter(x_train, y_train, x_test, y_test, p)
		rss[i] = rss0
	return np.mean(rss)


def rss(y, yhat):
	y = y.reshape(-1)
	yhat = yhat.reshape(-1)
	n = len(y)
	dev = np.subtract(y.astype('float_'), yhat.astype('float_'))
	return np.dot(dev, dev)

def mse(y, yhat):
	return rss(y, yhat) / len(y)

def errorRate(y, hat):
	n = len(y)
	F = 0
	for i in range(0,n):
		if y[i] != hat[i]:
			F = F + 1
	return F / n

def confusionMatrix(y, yHat):
	n = len(y)
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
	print(["tp", "fp", "tn" ,"fn"])
	print([tp, fp, tn ,fn])