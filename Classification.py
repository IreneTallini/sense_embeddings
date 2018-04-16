# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 15:57:17 2018

@author: irene
"""

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


import numpy as np
import os
from time import clock

from random import shuffle

#Load embeddings
embeddings = np.load('V3000E150W5B60N5I500000TRAIN/final_embeddings.npy').item()
len_emb = len(embeddings['UNK'])

#Read training dataset
train_directory = 'dataset/data/TRAIN'
stop_words = open('stop_words.txt', 'r').read().split('\n')[1:]
ascii_chars = {chr(i) for i in range(97, 123)}
dom_dict = {domain : index for index, domain in enumerate(os.listdir(train_directory))}
X = []
y = []
for domain in dom_dict:
    for index, file in enumerate(os.listdir(os.path.join(train_directory, domain))):
        if index < 10000:
            with open(os.path.join(train_directory, os.path.join(domain, file)), encoding = 'utf-8') as f: 
                f = f.read().split(' ')
                #Delete stop-words and characters not in [a,...,z]
                f = list(filter(lambda word : 
                                set([char for char in word]) & ascii_chars == set([char for char in word])
                                and not word in stop_words,
                                f))
                #Compute centroid of each file
                if len(f) != 0:
                    centroid = np.mean([embeddings[word] if word in embeddings 
                                        else np.zeros(len_emb) for word in f], axis = 0)
                    X.append(centroid)
                    y.append(dom_dict[domain])
        else:
            break

#Split in training and test set
shuffle(X)
shuffle(y)
train_len = int(0.5*len(X))
X_train = X[:train_len]
X_test = X[train_len:]
y_train = y[:train_len]
y_test = y[train_len:]

#Train kNN Classifier
model = KNeighborsClassifier(n_neighbors = 5).fit(X_train, y_train)
pred = model.predict(X_test)
print(classification_report(y_test, pred, target_names=dom_dict.keys()))

#Confusion Matrix
np.set_printoptions(threshold=np.nan)
fig, ax = plt.subplots(figsize=(8, 6))
bubu = ax.imshow(confusion_matrix(y_test, pred))
fig.colorbar(mappable=bubu)
tick_marks = np.arange(len(dom_dict))
plt.xticks(tick_marks, dom_dict, rotation=90)
plt.yticks(tick_marks, dom_dict)
plt.show()