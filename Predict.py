import nltk
import numpy as np
import gensim
from sklearn.externals import joblib
from scipy.spatial.distance import cosine

w2v = gensim.models.KeyedVectors.load_word2vec_format('Models/W2VBrown10_100')

def predictKNN(inText):
    tokens = nltk.word_tokenize(inText)
    vector = np.mean([w2v.wv[w] for w in tokens if w in w2v]
                     or [np.zeros(100)], axis=0)
    model = joblib.load('Models/KNNClassifier')
    prediction = int(model.predict([vector]).tolist()[0])
    probs = model.predict_proba([vector])[0].tolist()

    qfile = open("Data/Questions", "r")
    questions = []
    for line in qfile:
        questions.append(line)

    q_out = questions[prediction]
    tokens_out = nltk.word_tokenize(q_out)
    vector_out = np.mean([w2v.wv[w] for w in tokens_out if w in w2v]
                         or [np.zeros(100)], axis=0)
    similarity = 1 - cosine(vector, vector_out)
    if similarity > 0.9:
        final = q_out
    else:
        final = 'Unknown'

    classes = zip(probs,model.classes_)
    others = []
    for c in classes:
        if c[0] > 0 and c[1] != str(prediction):
            others.append(questions[int(c[1])])

    return prediction, probs, q_out, similarity, final, classes, others#, a_out

intext = 'who likes the color pink?'
prediction, probs, q_out, s, final, classes, others = predictKNN(intext)
print(others)
