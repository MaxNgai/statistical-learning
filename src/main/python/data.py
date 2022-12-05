import csv
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



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

class caravan:
	def __init__(self):
		enc = OrdinalEncoder()
		enc.fit(np.array(["Yes","No"]).reshape(-1,1))
		scaler = StandardScaler()
		self.raw = np.asarray(read("Caravan"))[1:]
		scaler.fit(self.raw[0:1000,0:85].astype('float_'))

		self.testX = scaler.transform(self.raw[0:1000,0:85].astype('float_'))
		self.testY = enc.transform(self.raw[0:1000,85].reshape(-1,1))
		self.trainX = scaler.transform(self.raw[1000:,0:85].astype('float_'))
		self.trainY = enc.transform(self.raw[1000:,85].reshape(-1,1))


class weekly:
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
		self.trainX = self.lag2[0:985]
		self.testX = self.lag2[985:1089]
		self.trainY = self.direction[0:985]
		self.testY = self.direction[985:1089]

class auto:
	def __init__(self):
		self.raw = np.asarray(read("Auto"))[1:]
		self.mpg = self.raw[..., 0].astype('float_')
		median = np.median(self.mpg)
		mpg0 = []
		for e in self.mpg:
			if e > median:
				mpg0.append(1)
			else:
				mpg0.append(0)
		self.mpg01 = mpg0
		for x in range(len(self.raw[...,3])):
			if self.raw[x,3] == "?": # the original data contain '?' which is not illegal for number predictors, so change to constant 100
				self.raw[x,3] = 100

		self.X1357 = np.vstack([self.raw[...,1].astype('float_'), self.raw[...,3].astype('float_'), self.raw[...,5].astype('float_'), self.raw[...,7].astype('float_')]).T
		self.testY = self.mpg01[:100]
		self.testX = self.X1357[:100,...]
		self.trainX = self.X1357[100:,...]
		self.trainY = self.mpg01[100:]

class hitters:
	def __init__(self):
		self.raw = np.asarray(read("Hitters"))[1:]
		toDelete = np.where(self.raw[...,18] == 'NA')
		self.Y = np.delete(self.raw[...,18], toDelete, axis = 0)
		self.X = np.delete(np.hstack([self.raw[...,:18], self.raw[...,19].reshape(-1,1)]), toDelete, axis = 0)
		

		leagueEnc = OrdinalEncoder()
		leagueEnc.fit(np.array(["A","N"]).reshape(-1,1))
		divisionEnc = OrdinalEncoder()
		divisionEnc.fit(np.array(["E","W"]).reshape(-1,1))
		
		self.X[...,18] = leagueEnc.transform(self.X[...,18].reshape(-1,1)).reshape(-1)
		self.X[...,13] = leagueEnc.transform(self.X[...,13].reshape(-1,1)).reshape(-1)
		self.X[...,14] = divisionEnc.transform(self.X[...,14].reshape(-1,1)).reshape(-1)
	
		scaler = StandardScaler()
		self.standarder = scaler
		scaler.fit(self.X.astype('float_'))
		self.X = scaler.transform(self.X)


class college:
	def __init__(self):
		self.raw = np.asarray(read("College"))[1:,1:]
		self.X = np.hstack([self.raw[...,0].reshape(-1,1), self.raw[..., 2:]])
		self.Y = self.raw[...,1].astype("float_")
		
		yesOrNo = OrdinalEncoder()
		yesOrNo.fit(np.array(["Yes", "No"]).reshape(-1,1))
		self.X[...,0] = yesOrNo.transform(self.X[...,0].reshape(-1,1)).reshape(-1)

		scaler = StandardScaler()
		scaler.fit(self.X.astype('float_'))
		self.X = scaler.transform(self.X).astype("float_")

		self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.X, self.Y, test_size = 0.5, random_state=66)
		self.train_x = self.train_x.astype("float_")
		self.train_y = self.train_y.astype("float_")
		self.test_x = self.test_x.astype("float_")
		self.test_y = self.test_y.astype("float_")

class boston:
	def __init__(self):
		self.raw = np.asarray(read("Boston"))[1:,1:]
		self.X = self.raw[..., 1:].astype("float_")
		self.Y = self.raw[..., 0].astype("float_")

		scaler = StandardScaler()
		scaler.fit(self.X.astype('float_'))
		self.X = scaler.transform(self.X)

		self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(self.X, self.Y, test_size = 0.1, random_state=77)
		self.train_x = self.train_x.astype("float_")
		self.train_y = self.train_y.astype("float_")
		self.test_x = self.test_x.astype("float_")
		self.test_y = self.test_y.astype("float_")




		

