#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import print_function

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

batch_size = 32
num_classes = 41
input_size = 10
classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
           11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
           21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 
           31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 
           41] 
class_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
    '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
    '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
    '31', '32', '33', '34', '35', '36', '37', '38', '39', '40',
    '41']
epochs = 500
learning_rate = 1e-5
early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=2)

lbp_train = pd.read_csv('../LBP_feature_train.csv')
lbp_val = pd.read_csv('../LBP_feature_val.csv')
train_label = pd.read_csv('../train_label.csv')
val_label = pd.read_csv('../val_label.csv')

# get your training and testing data here
# and to make sure to reshape and normalize!
#x_train = pd.concat([lbp_train, fd_train, color_som_train], axis=1).values
#x_test = pd.concat([lbp_val, fd_val, color_som_val], axis=1).values
x_train = lbp_train.values
x_test = lbp_val.values

# convert class vectors to binary class matrices (one-hot encoding)
# there are some missing class!

train_label_list = train_label['class_no'].tolist()
onehot_train = []
for value in train_label_list:
    class_id = [0 for _ in range(len(classes))]
    class_id[value - 1] = 1
    onehot_train.append(class_id)
y_train = pd.DataFrame(onehot_train)

val_label_list = val_label['class_no'].tolist()
onehot_val = []
for value in val_label_list:
    class_id = [0 for _ in range(len(classes))]
    class_id[value - 1] = 1
    onehot_val.append(class_id)
y_test = pd.DataFrame(onehot_val)

#X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=0)
X_train = x_train
X_val = x_test
Y_train = y_train
Y_val = y_test

# be sure of input layer!
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(input_size,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
    optimizer=Adam(lr=learning_rate),
    metrics=['accuracy'])

history = model.fit(X_train, Y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=2,
    validation_data=(X_val, Y_val),
    callbacks=[early_stopping])

print('test before save')
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1]) 
