import gensim, logging, os
from nltk.corpus import gutenberg, brown

sentences = brown.sents()
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# train word2vec on the two sentences
model = gensim.models.Word2Vec(sentences, min_count=100, size=100)
saveLocation=open('Models/W2VBrown10_100','w')
# model.save(saveLocation)
model.wv.save_word2vec_format(saveLocation)
model.accuracy('Data/assessment')
