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

'''
defaultData = default()
print(defaultData.income)
'''
