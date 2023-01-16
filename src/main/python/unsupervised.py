from scipy.cluster.hierarchy import dendrogram
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import data
import util
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

usarrests = data.usarrest()
nci = data.nci60()
gene = data.finalAssignment()

def plot_dendrogram(model, **kwargs):
        # Create linkage matrix and then plot the dendrogram

        # create the counts of samples under each node
        counts = np.zeros(model.children_.shape[0])
        n_samples = len(model.labels_)
        for i, merge in enumerate(model.children_):
           current_count = 0
           for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                   current_count += counts[child_idx - n_samples]
                   counts[i] = current_count

                   linkage_matrix = np.column_stack(
                       [model.children_, model.distances_, counts]
                       ).astype(float)

        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, **kwargs)
        plt.show()

class unsupervised:

    #p403
    def tryPca(self):
        scaler = StandardScaler()
        scaled = scaler.fit_transform(usarrests.X)
        
        pca = PCA()
        trans = pca.fit_transform(scaled)
        print(pca.components_) # n_component * n_feature

        print(trans[0]) # Z, pca score
        print(np.dot(scaled[0] , pca.components_.T)) # pca score = x * loading

        print(pca.explained_variance_ratio_)

        plt.scatter(trans[...,0], trans[...,1])
        v = pca.components_.T[..., :2]
        for i in v:
            plt.arrow(0,0,i[0],i[1])
        plt.show()

    #p404
    def tryKmeans(self):
        np.random.seed(1)
        x = np.random.randn(100).reshape(-1, 2)
        x[:25, 0] = x[:25, 0] + 3
        x[:25, 1] = x[:25, 1] - 4
        #util.scatter2d(x[:,0],x[:, 1])

        kmeans = KMeans(n_clusters = 2, n_init = 20)
        clustered = kmeans.fit_predict(x)
        print(clustered)

        for i in range(50):
            plt.scatter(x[i,0], x[i,1], c= "red" if i <25 else "blue")
        plt.show()

        kmeans3 = KMeans(n_clusters = 3, n_init = 20)
        clustered = kmeans3.fit_predict(x)
        print(clustered)

        for i in range(50):
            plt.scatter(x[i,0], x[i,1], c= "red" if clustered[i] ==0 else "blue" if clustered[i] ==1 else "green")
        plt.show()

    #p406
    def hierarchicalClustering(self):
        np.random.seed(1)
        x = np.random.randn(100).reshape(-1, 2)
        x[:25, 0] = x[:25, 0] + 3
        x[:25, 1] = x[:25, 1] - 4

        avg = AgglomerativeClustering(linkage='average').fit_predict(x)
        print(avg)
        single = AgglomerativeClustering(linkage='single').fit_predict(x)
        print(single)
        complete = AgglomerativeClustering(linkage='complete').fit_predict(x)
        print(complete)

    #p407
    def nci60(self):
        scaler = StandardScaler()
        scaled = scaler.fit_transform(nci.X)

        pca = PCA()
        trans = pca.fit_transform(scaled)
        
        util.plot(pca.explained_variance_ratio_) # elbow is the 7th
        #print(np.sum(pca.explained_variance_ratio_)) though m < p, yet sum = 1

        util.scatter2d(trans[:,0], trans[:,1])
        util.scatter2d(trans[:,0], trans[:,2])


    #p410
    def hierarchicalOnNic(self):
        scaler = StandardScaler()
        x = scaler.fit_transform(nci.X)

        avg = AgglomerativeClustering(linkage='average', distance_threshold=0, n_clusters = None).fit(x)
        #plot_dendrogram(avg, labels = nci.Y)
        single = AgglomerativeClustering(linkage='single', distance_threshold=0, n_clusters = None).fit(x)
        #plot_dendrogram(single, labels = nci.Y)
        complete = AgglomerativeClustering(linkage='complete', compute_distances=True, compute_full_tree=True).fit(x)
        plot_dendrogram(complete, labels = nci.Y, color_threshold=139) #color_threshold helps distinguish cluster, which is the height to cut 

    #p416-9
    def hierarchicalOnArrest(self):
        x = usarrests.X
        complete = AgglomerativeClustering(linkage='complete', compute_distances=True).fit(x)
        plot_dendrogram(complete, color_threshold=150, labels=usarrests.name) #color_threshold helps distinguish cluster, which is the height to cut 

    def hierarchicalOnArrestScaled(self):
        scaler = StandardScaler()
        x = scaler.fit_transform(usarrests.X)
        
        complete = AgglomerativeClustering(linkage='complete', compute_distances=True).fit(x)
        plot_dendrogram(complete, color_threshold=4.5, labels=usarrests.name) #color_threshold helps distinguish cluster, which is the height to cut 
        #completely different from unscaled. should scale because assault is large and will have heavy impact on euclidean distances.

    #p417-10
    def pcaAndKmeans(self):
        np.random.seed(1)
        x = np.random.randn(3000).reshape(-1, 50)
        y = np.hstack([np.ones(20) * 0, np.ones(20) * 1, np.ones(20) *2])
        meanShift = 0.5
        x[y==1,:] = x[y==1,:] + meanShift
        x[y==2,:] = x[y==2,:] - meanShift
        
        pca = PCA()
        trans = pca.fit_transform(x)
        self.seeSplit(trans[:, :2], y)    

        kmeans = KMeans(n_clusters = 3, n_init = 20)
        predict3 = kmeans.fit_predict(x)
        self.seeSplit(trans[:, :2], predict3) # bad

        kmeans2 = KMeans(n_clusters = 2, n_init = 20)
        predict2 = kmeans2.fit_predict(x)
        #self.seeSplit(trans[:, :2], predict2) # cluster[0] & cluster[1] merge into 1

        kmeans4 = KMeans(n_clusters = 4, n_init = 20)
        predict4 = kmeans4.fit_predict(x)
        #self.seeSplit(trans[:, :2], predict4) # cluster[0] mix up with cluster[3]

        kmeansPCA = KMeans(n_clusters = 3, n_init = 20)
        predictPCA = kmeansPCA.fit_predict(trans[:,:2])
        self.seeSplit(trans[:, :2], predictPCA) # same as using raw data. if apply meanShift = 0.5, there will be slight difference between two, but good enough.if meanShift = 3 then will be too wrong.
        # using pca will make it more non-overlaped within clusters

        scaler = StandardScaler()
        scaled = scaler.fit_transform(x)
        kmeansScale= KMeans(n_clusters = 3, n_init = 20)
        predictScale = kmeansScale.fit_predict(x)
        self.seeSplit(trans[:, :2], predictScale) # different but good enough, more accurate than kmeans on raw data
   

    
    def seeSplit(self, x, y):
        for i in range(len(y)):
            plt.scatter(x[i,0], x[i,1], c = "red" if y[i] == 0 else "green" if y[i] == 1 else "blue" if y[i] == 2 else "purple")
        plt.show()

    #p418-11
    def finalMission(self):
        scaler = StandardScaler()
        x = scaler.fit_transform( gene.X)
        color = np.asarray(list(map(lambda e: "red" if e==0 else "blue", gene.Y)))
        label = np.arange(0,1000,1).astype("str_")
        
        
        dist_matrix = 1 - np.corrcoef(x) # correlation-based distance        
        complete = AgglomerativeClustering(linkage='complete', compute_distances=True, affinity='precomputed').fit(dist_matrix)
        #plot_dendrogram(complete, color_threshold=1.1, labels=gene.Y) #correctly separated. any linkage will do

        pca = PCA()
        trans = pca.fit_transform(x)
        print(trans.shape)
        util.plot(pca.explained_variance_ratio_) # elbow is the 2th    
        plt.scatter(trans[:,0], trans[:,1], c=color) #well-separated
        plt.show()

        fpc = np.abs(pca.components_[0])
        maxIndex = np.where(fpc == np.max(fpc))
        print(fpc[maxIndex]) #gene[501] has largest loading

        vectors = trans[:2, :].T
        plt.scatter(vectors[:,0], vectors[:,1], label = label) #well-separated
        plt.show()
        print(vectors[np.where(vectors[:,0] > 15)])









unsupervised().finalMission()