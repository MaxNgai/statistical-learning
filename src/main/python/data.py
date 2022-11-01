import csv
import numpy as np
from sklearn.preprocessing import OrdinalEncoder


def getFilePath(fileName):
		return "../resources/ALL CSV FILES - 2nd Edition/" + fileName + ".csv"

def read(fileName):
	path = getFilePath(fileName)
	res = []

	file = csv.reader(path)
	

	with open(path) as f:
		header = next(file)

		for row in csv.reader(f):
			res.append(row)

	return res


	

class default:

	def __init__(self):
		enc = OrdinalEncoder()
		enc.fit(np.array(["No","Yes"]).reshape(-1,1))
		self.raw = np.asarray(read("Default"))[1:]
		self.balance = self.raw[...,2].reshape(-1,1).astype('float_')
		self.default = self.raw[...,0]
		self.income = self.raw[...,3].reshape(-1,1).astype('float_')
		self.isStudent = enc.transform(self.raw[...,1].reshape(-1,1))
		self.X = np.hstack((self.isStudent, self.balance))


class smarket:
	def __init__(self):
		enc = OrdinalEncoder()
		enc.fit(np.array(["Up","Down"]).reshape(-1,1))
		self.raw = np.asarray(read("Smarket"))[1:]
		self.year = self.raw[...,0].reshape(-1,1).astype('int')
		self.lag1 = self.raw[...,1].reshape(-1,1).astype('float_')
		self.lag2 = self.raw[...,2].reshape(-1,1).astype('float_')
		self.lag3 = self.raw[...,3].reshape(-1,1).astype('float_')
		self.lag4 = self.raw[...,4].reshape(-1,1).astype('float_')
		self.lag5 = self.raw[...,5].reshape(-1,1).astype('float_')
		self.volume = self.raw[...,6].reshape(-1,1).astype('float_')
		self.today = self.raw[...,7].reshape(-1,1).astype('float_')
		self.direction = enc.transform(self.raw[...,8].reshape(-1,1))
		self.volumeAnd1to5 = np.hstack([self.lag1,self.lag2,self.lag3,self.lag4,self.lag5,self.volume])

