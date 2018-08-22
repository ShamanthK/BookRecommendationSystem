#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 21:36:25 2018

@author: lelf
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 20:51:38 2018

@author: lelf
"""

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import csv
import random
from scipy.spatial import distance


randrating = []
randcluster = []
euclidean_distance = []
cluster_rating = []
fittnes_score = []
data = pd.read_csv('books.csv')
print(data.shape)
data.head()
  
f1 = data['book_id'].values
f2 = data['average_rating'].values
f3 = data['original_title'].values
X = np.array(list(zip(f2)))
    
NK = 65
# Number of clusters
kmeans = KMeans(n_clusters=NK)
# Fitting the input data
kmeans = kmeans.fit(X)
# Getting the cluster labels
labels = kmeans.predict(X)
# Centroid values
centroids = kmeans.cluster_centers_
print(centroids)

X1 = np.array(list(zip(f1, f2, f3, labels)))  


with open('Cluster.csv','w') as f:
    w = csv.writer(f)
    w.writerow(['Ratings','Title', 'Labels'])
    w.writerows(X1)
    
  
b_ID = 1569
K = f1.size

for i in range(K):
    if(b_ID == f1[(i)]):
        rating = f2[(i)]
        title = f3[(i)]
        label = labels[(i)]   
   
           


#cuckoo search
rand_rat = round(random.uniform(3,5),2)
rand_clu = random.randint(1,65)
print (rand_rat, rand_clu)
for i in range(K):
    if(rand_clu == labels[(i)]):
        clu_rating = f2[(i)]
ed0 = distance.euclidean(rand_rat,clu_rating)
euclidean_distance.append([ed0])
fittnes_score.append([ed0])


for k in range(500):
    rand_rat = round(random.uniform(3,5),2)
    randrating.append([rand_rat])
    rand_clu = random.randint(1,65)
    randcluster.append([rand_clu])
    print (rand_rat, rand_clu)
    for i in range(K):
        if(rand_clu == labels[(i)]):
            clu_rating = f2[(i)]
            cluster_rating.append([clu_rating])
    ed1 = distance.euclidean(rand_rat,clu_rating)
    euclidean_distance.append([ed1])
    Ed = np.asarray(euclidean_distance)
    F = Ed[(k)] - Ed[(k+1)]
    fittnes_score.append([F])
    FS = np.asarray(fittnes_score)
    if(FS[(k)]>FS[(k+1)]):
        cs_cluster = rand_clu
        cs_rating = clu_rating
    else: 
        NK = NK+1

data1 = pd.read_csv('Cluster.csv')
data1.head()
X2 = data1.loc[data1['Labels'] == cs_cluster]

print 'Input book'
print(title)     
print 'Recommendation'
a1 = X2['Title'].values
print(a1[0:5])        