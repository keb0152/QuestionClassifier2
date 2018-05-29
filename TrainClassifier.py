import numpy as np
import gensim, nltk, pickle
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.externals import joblib

print("reading data...")
inFile = open('Data/variants','r')
print("loading word2vec...")
w2v = gensim.models.KeyedVectors.load_word2vec_format('Models/W2VBrown10_100')

print("vectorizing...")
x = []
y = []
for line in inFile:
    text, label = line.split('\t')
    label = label.strip('\n')
    tokens = nltk.word_tokenize(text)
    vector = np.mean([w2v.wv[w] for w in tokens if w in w2v]
                     or [np.zeros(100)], axis=0)
    x.append(vector)
    y.append(label)

print("training...")
newmodel = KNN(n_neighbors=5)
newmodel.fit(x,y)

joblib.dump(newmodel, 'Models/KNNClassifier')
