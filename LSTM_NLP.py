# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 13:49:09 2017
@author: Faber Vincent
Credits: https://github.com/tflearn/tflearn/blob/master/examples/nlp/lstm.py and Stack Overflow among other
"""

###############################################################################
#some imports
from collections import Counter
import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
import numpy as np
import sklearn
import re 
import operator
import pandas as pd
from sklearn.model_selection import train_test_split
###############################################################################

###############################################################################
#load/process Cons comments
f = open('IntegratedCons.txt','r')
all_text = f.read()
f.close()

#delete some special characters / bad strings
for elem in ['</Cons>\n', '\n', '  ', '\'']:
	all_text = all_text.replace(elem, '')

#lower case
all_text = all_text.lower()

#split on <cons>
mylist = all_text.split('<cons>')

#remove punctuation and tokenize (add 0 to tuple to encode 'bad review')
cons_list = []
for review in mylist:
	cons_list.append((re.sub("[^\w]", " ",  review).split(), 0))
###############################################################################

###############################################################################
#load/process Pros comments
f = open('IntegratedPros.txt','r')
all_text = f.read()
f.close()

#delete some special characters / bad strings
for elem in ['</Pros>\n', '\n', '  ', '\'']:
	all_text = all_text.replace(elem, '')

#lower case
all_text = all_text.lower()

#split on <cons>
mylist = all_text.split('<pros>')

#remove punctuation and tokenize (add 1 to tuple to encode 'good review')
pros_list = []
for review in mylist:
	pros_list.append((re.sub("[^\w]", " ",  review).split(), 1))
###############################################################################


###############################################################################
#all comments (pros and cons ones)
all_comments = pros_list + cons_list

#store in pd df
df = pd.DataFrame()
df['X'], df['Y'] = zip(*all_comments)
###############################################################################


###############################################################################
#this snippet computes overall word distribution (for fun)
flattened_comments = [word for comment in list(df['X']) for word in comment]
word_freq = dict(Counter(flattened_comments))
word_freq = sorted(word_freq.items(), key=operator.itemgetter(1), reverse = True) #list of tuples

unique_words = list(set(flattened_comments))
###############################################################################

###############################################################################
#define initial word mapping (THERE IS NO 0 INDEX! RESERVED FOR PADDING...)
word_to_idx = {}
idx_to_word = {}
i=1 #0 means 'no word'
for word in unique_words:
	word_to_idx[word] = i
	idx_to_word[i] = word
	i=i+1
###############################################################################

###############################################################################
#encode reviews
def encode_review(review):
	encoded_review = []
	for word in review:
		encoded_review.append(word_to_idx[word])
	return encoded_review
df['X'] = list(map(encode_review, df['X']))
###############################################################################	
	

###############################################################################
#parameters
VOCAB_SIZE = len(unique_words) + 1 #0 means 'no word'
MAX_LEN = 14 #maximum number of words to be considered in a sentence
EMBEDDING_DIM = 10 #number of dimension in the word embedding space 
BATCH_SIZE = 100
###############################################################################


###############################################################################
#add padd (0's) for same seq len (when a given comment has less than MAX_LEN number of words)
df['X'] = list(pad_sequences(list(df['X']), maxlen=MAX_LEN, value=0.))
###############################################################################
	
###############################################################################
# Converting labels to binary vectors
df['Y'] = list(to_categorical(list(df['Y']), nb_classes = 2))
###############################################################################
	
###############################################################################
#to print encoded reviews...
def padded_vect_to_review(x):
	out = []
	j = 0
	while x[j] != 0:
		out.append(idx_to_word[x[j]])
		j = j + 1
	return out
#e.g.
print('\'{}\' decodes to: \n{}'.format(df['X'][2], padded_vect_to_review(df['X'][2])))
###############################################################################

###############################################################################
#split train/test
train, test = train_test_split(df, train_size = 0.67, test_size = 0.33)
trainX, trainY = np.array(list(train['X'])), np.array(list(train['Y']))
testX, testY = np.array(list(test['X'])), np.array(list(test['Y']))
###############################################################################





###############################################################################
#build the lstm net
tf.reset_default_graph()

#defines the input tensor's dimensionality: [batch_size, input_vector_length]
input_layer = tflearn.input_data([None, MAX_LEN])

#notes about word embedding layer:
#	    This layer acts as a word embedding (e.g. same concept as word2vec) algorithm. 
#	    Word2vec works as a 3 layer neural net: 1 input layer + 1 hidden layer + 1 output layer
#	    Given a word, it's trained to predict its neighbouring words. Once trained, one can remove the last (output layer) and keep the input and hidden layer.
#	    Now, feeding in an input word from within the vocabulary, the output given at the hidden layer corresponds to the word 'embedding'.
#	    A famous example: given some text input field has "queen", "king", "girl","boy" and 2 embedding dimensions, hopefully the backpropagation will train the embedding to 
#	    put the concept of royalty on one axis and gender on the other. In this case the vocab size (aka starting dimension) is 4 (actually 5 if we include 'no word') 
#	    and the words get mapped to a 2 dimensional space.
#	    For instance, a sentence [1, 4, 2, 0, 0] (Note that here the MAX_LEN is 5 and we must zero pad the vector with 2 extra 0's), which happens to have 3 words will be parsed
#	    (for output_dim=2) to something like [[0.0, 1.2], [2.5, 4.9], [2.0, 5.2], [0, 0], [0, 0]].
#	    This sequence is what is used as input for the lstm layer. By definition, the sequence length will corresponds to the MAX_LEN.
#	    Note that first, before embedding, each word is implicitely 1-hot encoded (e.g. queen --> [1, 0, 0, 0, 0] and king --> [0, 1, 0, 0, 0])

#defines the embedding layer
#warning: input_dim does NOT correspond to the output shape of the previous layer. It has to be the VOCAB_SIZE, so that any word (once 1-hot encoded) can be passed through this layer
#output_dim corresponds to the number of hidden units i.e. the dimensionality of the embedded word vector
embedding_layer = tflearn.embedding(input_layer, input_dim = VOCAB_SIZE, output_dim = EMBEDDING_DIM)

#defines the lstm layer. Takes as input [batch_size, timesteps (MAX_LEN), data_dimension (EMBEDDING_DIM)]
#The output is a tensor with dimension [batch_size, n_units]
lstm_layer = tflearn.lstm(embedding_layer, n_units=100, dropout=0.8, return_seq = False, weights_init = 'normal')  

#defines fully connected layer with 2 hidden units and a softmax activation
output_layer = tflearn.fully_connected(lstm_layer, 2, activation='softmax', weights_init = 'normal')

#compute loss, gradient, etc.
loss = tflearn.regression(output_layer, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy')

#training
model = tflearn.DNN(loss, tensorboard_verbose=0)
model.fit(trainX, trainY, show_metric = True, batch_size = BATCH_SIZE , n_epoch = 1)
###############################################################################

###############################################################################
#get auc
auc = sklearn.metrics.roc_auc_score(testY, model.predict(testX), average="macro", sample_weight=None)
#get accuracy
pred = model.predict(testX).argmax(axis=1)
def compare(a,b):
	if a==b:
		return 1
	else:
		return 0	
accuracy = np.array(list(map(compare, pred, testY.argmax(axis=1)))).mean()
print('Accuracy on test set: {} - AUC on test set: {}'.format(accuracy, auc))
###############################################################################

###############################################################################
#fun test
p=10000
print('reviews: {}'.format(padded_vect_to_review(testX[p])))
print('true: {}'.format(testY[p]))
print('pred: {}'.format(model.predict([testX[p]])))
###############################################################################

###############################################################################
#get the embedding layer weights. It corresponds to your mapping from VOCABULARY_SIZE to EMBEDDING_DIM
w2v = model.get_weights(embedding_layer.W)

#e.g. embedding 
word = np.zeros((VOCAB_SIZE,1))
word[2] = 1
v1 = np.matmul(w2v.T, word)

#another example
def get_embedding(word):
	temp = np.zeros((VOCAB_SIZE,1))
	temp[word_to_idx[word]] = 1
	return np.matmul(w2v.T, temp)

get_embedding('bad').reshape((EMBEDDING_DIM))
###############################################################################



