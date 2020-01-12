# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 06:56:46 2020

@author: ASUS
"""

from IPython import get_ipython
def __reset__(): get_ipython().magic('reset -sf')

__reset__()

#-----------------------------------------------------------------------------#

import numpy as np
import keras.preprocessing
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout
from keras.layers import LSTM

#-----------------------------------------------------------------------------#

datafile_pos = 'data/train-pos.txt'
datafile_neg = 'data/train-neg.txt'
datafile_eval = 'data/evaluation.txt'
datafile_out = 'predictions.txt'

dataset_pos = []
dataset_neg = []
dataset_eval = []
dataset_eval_text = []

data = []
labels = []

x_train = []
y_train = []

x_test = []
y_test = []

classes = []
out = []

vocab_size = 10000
max_length = 100
batch_size = 128

#-----------------------------------------------------------------------------#

def rev_split(datafile, dataset):

    with open(datafile, 'r') as f:
        dataset = f.readlines()                    
    return dataset

def one_hot(dataset, vocab_size):
    
    onehotset = []
    
    for i in dataset:
        onehotset.append(keras.preprocessing.text.one_hot(i, vocab_size, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' '))
    return onehotset

def data_shuffle(dataset_pos, dataset_neg, data_shuffled, labels_shuffled):
    
    i = 0

    while i < len(dataset_pos):
        data_shuffled.append(dataset_pos[i])
        data_shuffled.append(dataset_neg[i])
        labels_shuffled.append(1)
        labels_shuffled.append(0)
        i += 1
        
    i = 0
    
    return data_shuffled, labels_shuffled


def data_split(xtrain, ytrain, xtest, ytest, data, labels):
    
    length = len(data)
    train_length = int(len(data) * 0.8)
    i = 0
    
    while i < length:
        if i < train_length: 
            xtrain.append(data[i])
            ytrain.append(labels[i])
            i += 1
        else:
            xtest.append(data[i])
            ytest.append(labels[i])
            i += 1
    
    i = 0
    length = 0
    train_length = 0
            
    return xtrain, ytrain, xtest, ytest

#-----------------------------------------------------------------------------#

print("Uploading data...\n")

#Splitting reviews
dataset_pos = rev_split(datafile_pos, dataset_pos)
dataset_neg = rev_split(datafile_neg, dataset_neg)
dataset_eval = rev_split(datafile_eval, dataset_eval)
dataset_eval_text = rev_split(datafile_eval, dataset_eval)

print("Total reviews: "+str(len(dataset_pos)+len(dataset_neg)))
print("Number of negative reviews: "+str(len(dataset_neg)))
print("Number of positive reviews: "+str(len(dataset_pos)))
print("Total test reviews: "+str(len(dataset_eval)))


print("\nPreprocessing...")

#One-hot-encoding reviews
dataset_pos = one_hot(dataset_pos, vocab_size)
dataset_neg = one_hot(dataset_neg, vocab_size)
dataset_eval = one_hot(dataset_eval, vocab_size)

#Shuffling data
data_shuffle(dataset_pos, dataset_neg, data, labels)

#Splitting preliminary train and test data
data_split(x_train, y_train, x_test, y_test, data, labels)

#Padding
x_train = sequence.pad_sequences(x_train, maxlen=max_length, padding = 'post')
x_test = sequence.pad_sequences(x_test, maxlen=max_length, padding = 'post')
dataset_eval = sequence.pad_sequences(dataset_eval, maxlen=max_length, padding = 'post')
print('x_train shape:', x_train.shape)
#print('x_test shape:', x_test.shape)

#ML Model
print('\nBuilding model...')

model = Sequential()
model.add(Embedding(vocab_size, 128, input_length = max_length))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
#model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
    
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())

print('Training...')
model.fit(x_train, y_train,
          batch_size = batch_size,
          epochs = 5,
          shuffle = True,
          validation_data = (x_test, y_test))
loss, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test accuracy:', acc)

classes = model.predict(dataset_eval, batch_size = batch_size)

for i in classes:
    if float(i) >= 0.5:
        out.append(1)
    elif float(i) < 0.5:
        out.append(0)
#%%
with open('datafile_out.txt', 'w') as f:
    for line in out:
        f.write(str(line))
        f.write("\n")
    
        
