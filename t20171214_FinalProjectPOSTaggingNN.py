# -*- coding: utf-8 -*-
from nltk.tag.sequential import UnigramTagger
from nltk.corpus import brown
from nltk.tokenize import TreebankWordTokenizer
from nltk import tokenize
import numpy as np
from gensim.models import word2vec
import keras
import keras.preprocessing.sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.layers.core import Dense, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers import Convolution1D, Bidirectional
from keras.models import load_model

allCats=brown.categories()
numSamples=1000 #Change this
allsents=brown.tagged_sents(categories=allCats)[0:numSamples]
allsentsTest=brown.tagged_sents(categories=allCats)[1000:1100]
allsentsCleaned=[]
allsentsTestCleaned=[]
#Clean the labels to get rid of the TL
for i in range(0,len(allsents)):
    newSent=[]
    curSent=allsents[i]
    for word in curSent:
        newSent.append((word[0],word[1].split("-")[0]))
    allsentsCleaned.append(newSent)
for i in range(0,len(allsentsTest)):
    newSent=[]
    curSent=allsentsTest[i]
    for word in curSent:
        newSent.append((word[0],word[1].split("-")[0]))
    allsentsTestCleaned.append(newSent)
allLabels=[]
allWords=[]
for sent in allsents:
    for word in sent:
        allWords.append(word[0].lower())
        allLabels.append(word[1].split("-")[0])
uniqueWords=np.unique(allWords)
uniqueLabels=np.unique(allLabels)
print("num seqs:")
print(len(allsents))
print("num labeled words:")
print(len(allWords))


#################################Testing code##################################
######HMM TESTING##########
#testLabelsHMM=[]
#for i in range(0, len(allsentsTest)):
#    mystr=[]
#    for j in range(0,len(allsentsTest[i])):
#        mystr.append(allsentsTest[i][j][0].lower())
#    myout=forwardAlgo(mystr, transitionCounts, posCounts, uniqueWords, uniqueLabels, stateProbs)
#    testLabelsHMM.append(myout)
#numCorrectHMM=0
#numIncorrectHMM=0
#numCorrectUnknown=0
#numIncorrectUnknown=0
#numCorrectKnown=0
#numIncorrectKnown=0
#for i in range(0,len(testLabelsHMM)):
#    for j in range(0,len(testLabelsHMM[i])):
#        if testLabelsHMM[i][j]==allsentsTest[i][j][1]:
#            numCorrectHMM=numCorrectHMM+1
#            if allsentsTest[i][j][0] in uniqueWords:
#                numCorrectKnown=numCorrectKnown+1
#            else:
#                numCorrectUnknown=numCorrectUnknown+1
#        else:
#            numIncorrectHMM=numIncorrectHMM+1
#            if allsentsTest[i][j][0] in uniqueWords:
#                numIncorrectKnown=numIncorrectKnown+1
#            else:
#                numIncorrectUnknown=numIncorrectUnknown+1
#            
#print("HMM score")
#print(numCorrectHMM/float(numCorrectHMM+numIncorrectHMM))
#print("HMM Known score")
#print(numCorrectKnown/float(numCorrectKnown+numIncorrectKnown))
#print("HMM UnKnown score")
#print(numCorrectUnknown/float(numCorrectUnknown+numIncorrectUnknown))

#################################################################
##Neural network experiments
#figure out max sequence length in training


trainedNetworkDir='./model/'
modelName='testModel2.h5'
allLens=[]
for sent in allsentsCleaned:
    allLens.append(len(sent))
print("Max sequence len :"+str(np.max(allLens)))

allLens=[]
for sent in allsentsTest:
    allLens.append(len(sent))
def makeW2Vmodel(inputSents):
    tokenizedSents=[]
    for i in range(0,len(inputSents)):
        curSent=inputSents[i]
        curTokenizedSent=[]
        for word in curSent:
            curTokenizedSent.append(word[0].lower())
        tokenizedSents.append(curTokenizedSent)
    #Now that we have the training strings, we need to make the word embeddings
    w2vModel=word2vec.Word2Vec(tokenizedSents, iter=200, sg=0)
    vocab=list(w2vModel.wv.wv.vocab.keys())
    return w2vModel, vocab
def convertToEmbedInts(inputSents, vocab, maxLen):
    embedInts=[]
    for i in range(0,len(inputSents)):
        curSent=inputSents[i]
        embedding=[]
        for word in curSent:
            curWord=word[0].lower()
            if curWord in vocab:
                embedding.append(vocab.index(curWord)+1)
            else:
                embedding.append(0)
        embedInts.append(keras.preprocessing.sequence.pad_sequences([embedding], maxlen=maxLen)[0])
    return np.asarray(embedInts)
def convertToBinaryLabels(inputSents, possLabels, maxLen):
    binaryLabels=[]
    for i in range(0,len(inputSents)):
        curBinaryLabel=np.zeros((maxLen,len(possLabels)))
        #Set default label to be 0 for all indicies
        curBinaryLabel[:,0]=1
        curSent=inputSents[i]
        for j in range(0,len(curSent)):
            #Fill in from the end
            curLabel=curSent[-1*j][1]
            if curLabel in possLabels:
                curInd=list(possLabels).index(curLabel)
                curBinaryLabel[-1*j,curInd]=1
        binaryLabels.append(curBinaryLabel)
    return np.asarray(binaryLabels)

def makeEmbeddingMatrix(model, vocab):
    mykeys=vocab
    embeddingMatrix=np.zeros((len(mykeys)+1,model.vector_size))
    ind=1 #First entry is all zeros
    for key in mykeys:
        embeddingMatrix[ind]=model[key]
        ind=ind+1
    return embeddingMatrix

maxLen=75
curModel, curVocab=makeW2Vmodel(allsentsCleaned)
embedMat=makeEmbeddingMatrix(curModel,curVocab)
myints=convertToEmbedInts(allsents,curVocab,maxLen)
mylabels=convertToBinaryLabels(allsents,uniqueLabels, maxLen)

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.layers.core import Dense, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers import Convolution1D, Bidirectional
from keras.models import load_model
from keras.callbacks import ModelCheckpoint


model = Sequential()
model.add(Embedding(np.shape(embedMat)[0], np.shape(embedMat)[1],input_length=maxLen,weights=[embedMat],trainable=False))
model.add(LSTM(256,return_sequences=True))
#model.add(LSTM(256, return_sequences=True))
#model.add(TimeDistributed(Dense(128, activation='tanh')))
model.add(TimeDistributed(Dense(len(uniqueLabels), activation='ReLU')))#Make this adapt to ds
model.add(Dense(len(uniqueLabels),activation='sigmoid'))
adam = optimizers.adam(lr=0.001)
model.compile( 'categorical_crossentropy', metrics=['accuracy'],optimizers=adam)

numTrain=800
x_train=myints[0:numTrain,:]
y_train=mylabels[0:numTrain,:,:]
x_val=myints[numTrain:,:]
y_val=mylabels[numTrain:,:,:]
checkpointer = ModelCheckpoint(filepath=trainedNetworkDir+modelName, verbose=1, save_best_only=True)
model.fit(x_train, y_train, batch_size=128, epochs=200, verbose=1, validation_data=(x_val,y_val), callbacks=[checkpointer])
model=load_model(trainedNetworkDir+modelName)
#Predict sentences

myintsTest=convertToEmbedInts(allsentsTest,curVocab,maxLen)
predictions=model.predict(myintsTest)

def convertPredictionsToLabels(inputSents,inputPredictions, possLabels):
    #Need input sents to get the correct length
    outLabs=[]
    for i in range(0,len(inputSents)):
        curLen=len(inputSents[i])
        curPred=predictions[i,-1*curLen:,:]
        curSentLabs=[]
        for j in range(0, curLen):
            curSentLabs.append(possLabels[np.argmax(curPred[j,:])])
        outLabs.append(curSentLabs)
    return outLabs







