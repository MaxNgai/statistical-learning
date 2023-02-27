import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

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
