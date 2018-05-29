import nltk
from nltk.corpus import wordnet as wn
from itertools import product
import json

def getTokens(text):
    tokenize = nltk.word_tokenize(text)
    tags = nltk.pos_tag(tokenize)
    return tags

def synonymsVB(token):
    synonyms = []
    synset = wn.synsets(token, pos = wn.VERB)
    if synset != []:
        topSense = synset[0]
        lemmas = topSense.lemmas()
        for l in lemmas:
            if l.name() not in synonyms:
                synonyms.append(l.name())
    return(synonyms)

def synonymsNN(token):
    synonyms = []
    synset = wn.synsets(token, pos=wn.NOUN)
    if synset != []:
        topSense = synset[0]
        lemmas = topSense.lemmas()
        for l in lemmas:
            if l.name() not in synonyms:
                synonyms.append(l.name())
    return(synonyms)

def sentenceVars(taggedTerms):
    synsent = []
    for taggedToken in taggedTerms:
        if taggedToken[1][0] == 'N': #and taggedToken[1] != 'NNP':
            variants = synonymsNN(taggedToken[0])
            if len(variants) != 0:
                synsent.append(variants)
            else:
                synsent.append([taggedToken[0]])
        elif taggedToken[1][0] =='V':
            variants = synonymsVB(taggedToken[0])
            if len(variants) != 0:
                synsent.append(variants)
            else:
                synsent.append([taggedToken[0]])
        else:
            synsent.append([taggedToken[0]])
    return synsent

def toString(synSent):
    out = []
    for item in synSent:
        sentence = " ".join(item)
        out.append(sentence)
    return out

def makeVariantsFile(infile, outfile):
    print("reading questions...")
    inSents = open(infile,"r")
    allSentences = []
    print("getting variants...")
    for i, sentence in enumerate(inSents):
        tokenize = nltk.word_tokenize(sentence)
        tags = nltk.pos_tag(tokenize)
        synonymSets = sentenceVars(tags)
        out1 = list(product(*synonymSets))
        out2 = toString(out1)
        for item in out2:
            allSentences.append([item, i])

    print("writing file...")
    fileout = open(outfile, 'w')
    for line in allSentences:
        fileout.write(str(line[0]))
        fileout.write('\t')
        fileout.write(str(line[1]))
        fileout.write('\n')
    return

makeVariantsFile('Data/Questions', 'Data/Variants')