import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering



#from sklearn.preprocessing import StandardScaler
#from sklearn.cluster import KMeans
def kMeansClustering(x):
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    model = KMeans(n_clusters = 2, n_init = 20)
    clustered = kmeans.fit_predict(x)

#from sklearn.cluster import AgglomerativeClustering
#from sklearn.preprocessing import StandardScaler
def hierarchicalClustering(x, label):
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # distance_threshold set the line between merge and not merge, finding final clusters
    model = AgglomerativeClustering(linkage='average',distance_threshold=0, n_clusters = None)
    clustered = model.fit_predict(x)

    plot_dendrogram(model, color_threshold=4.5, labels=label)

#import matplotlib.pyplot as plt
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