from sklearn.model_selection import train_test_split
import numpy as np


def cv(x, y, rssGetter):
	k = 10
	rss = np.empty([k])
	for i in range(k):
		x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
		rss0 = rssGetter(x_train, y_train, x_test, y_test)
		rss[i] = rss0
	return np.mean(rss)


def rss(y, yhat):
	y = y.reshape(-1)
	yhat = yhat.reshape(-1)
	n = len(y)
	dev = np.subtract(y.astype('float_'), yhat.astype('float_'))
	return np.dot(dev, dev)

def mse(y, hat):
	return rss(y, yhat) / len(y)