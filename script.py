-*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

# #%%
# from IPython import get_ipython
# def __reset__(): get_ipython().magic('reset -sf')

# __reset__()

# #Reset all variables cuz my IDE is dumb

import math
import numpy as np
import keras.preprocessing

#%%
datafile_pos = 'train-pos.txt'
datafile_neg = 'train-neg.txt'

dataset_pos = []
dataset_neg = []

data = []
labels = []

x_train = []
y_train = []

x_test = []
y_test = []

#Here we will store all the processed reviews

def rev_split(datafile, dataset):

    with open(datafile, 'r') as f:
        datalist = f.readlines()
        c = ''
        for i in datalist:
            for j in i:
                if j != '.':
                    c = c + j
                elif j == '.' and c != '':
                    dataset.append(c)
                    c = ''
                else:
                    pass
                    
    return dataset

#Here we split the massive chaotic entropic messy review file into individual reviews

def char_filter(dataset):
    
    for i in dataset:
        index = dataset.index(i)
        dataset[index] = keras.preprocessing.text.text_to_word_sequence(dataset[index], filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=False, split=' ')
    return dataset

#Here we filter all the unnecessary characters
    
def data_shuffle(data_pos, data_neg, data_shuffled, labels_shuffled, length):
    
    i = 0
    
    while i < length:
        data_shuffled.append(data_pos[i])
        data_shuffled.append(data_neg[i])
        labels_shuffled.append(1)
        labels_shuffled.append(0)
        i += 1
    return data_shuffled, labels_shuffled

def data_split(xtrain, ytrain, xtest, ytest, data, labels, length):
    
    train_length = int(length * 0.8)
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
            
    return xtrain, ytrain, xtest, ytest

#%%
max_length = 100
#%%
print("Uploading data...")

rev_split(datafile_pos, dataset_pos)
rev_split(datafile_neg, dataset_neg)

#%%
print("Preprocessing...")

char_filter(dataset_pos)
char_filter(dataset_neg)

#%%
print("Shuffling...")

data_shuffle(dataset_pos, dataset_neg, data, labels, max_length)

#%%
print("Splitting...")

data_split(x_train, y_train, x_test, y_test, data, labels, max_length)
print("---------- TRAIN DATA ----------\n\n")

i = 0

while i < len(x_train):
    print(str(i)+"\t"+str(x_train[i])+"\t"+str(y_train[i]))
    i += 1
    
i = 0

print("---------- TEST DATA ----------\n\n")

i = 0

while i < len(x_test):
    print(str(i)+"\t"+str(x_test[i])+"\t"+str(y_test[i]))
    i += 1
    



