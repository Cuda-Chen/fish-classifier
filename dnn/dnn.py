#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import print_function

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

import pandas as pd
from sklearn.preprocessing import LabelEncoder

batch = 32
num_classes = 42 # 41 + 1
class_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 
    '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
    '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
    '31', '32', '33', '34', '35', '36', '37', '38', '39', '40',
    '41', '42']
epochs = 100

lbp_train = pd.read_csv('../LBP_feature_train.csv')
lbp_val = pd.read_csv('../LBP_feature_val.csv')
color_som_train = pd.read_csv('../color_som_train.csv')
color_som_val = pd.read_csv('../color_som_val.csv')
train_label = pd.read_csv('../train_label.csv')
val_label = pd.read_csv('../val_label.csv')

# get your training and testing data here
# and to make sure to reshape and normalize!
x_train = pd.concat([lbp_train, color_som_train], axis=1).values
x_test = pd.concat([lbp_val, color_som_val], axis=1).values

# convert class vectors to binary class matrices (one-hot encoding)
# there are some missing class!
encoder = LabelEncoder()
classes_onehot = encoder.
y_train
y_test

# be sure of input layer!
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(58,)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
    optimizer=Adam(lr=1e-5),
    metrics=['accuracy'])

history = model.fit(x_train, y_train,
    batch=batch,
    epochs=epochs,
    verbose=1,
    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
