#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import print_function

import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input, add
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import IPython
import matplotlib.pyplot as plt

from numpy.random import seed
seed(1333)

batch_size = 32
num_classes = 41
lbp_input_size = 10
fd_input_size = 63
color_som_input_size = 48
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
epochs = 20000
learning_rate = 1e-5
early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=2)

lbp_train = pd.read_csv('../LBP_feature_train.csv')
lbp_val = pd.read_csv('../LBP_feature_val.csv')
fd_train = pd.read_csv('../fd_feature_train.csv')
fd_val =  pd.read_csv('../fd_feature_val.csv')
color_som_train = pd.read_csv('../color_som_train.csv')
color_som_val = pd.read_csv('../color_som_val.csv')
train_label = pd.read_csv('../train_label.csv')
val_label = pd.read_csv('../val_label.csv')

# normalize each feature
lbp_train_norm = (lbp_train - lbp_train.min()) / (lbp_train.max() - lbp_train.min())
color_som_train_norm = (color_som_train - color_som_train.min()) / (color_som_train.max() - color_som_train.min())
fd_train_norm = (fd_train - fd_train.min()) / (fd_train.max() - fd_train.min())
lbp_val_norm = (lbp_val - lbp_val.min()) / (lbp_val.max() - lbp_val.min())
color_som_val_norm = (color_som_val - color_som_val.min()) / (color_som_val.max() - color_som_val.min())
fd_val_norm = (fd_val - fd_val.min()) / (fd_val.max() - fd_val.min())

# drop fd_0 column because it is NaN
fd_train_norm = fd_train_norm.drop(columns="fd_0")
fd_val_norm = fd_val_norm.drop(columns="fd_0")


# get your training and testing data here
# and to make sure to reshape and normalize!
#x_train = pd.concat([lbp_train, fd_train, color_som_train], axis=1).values
#x_test = pd.concat([lbp_val, fd_val, color_som_val], axis=1).values
'''
x_lbp_train = lbp_train_norm.values
x_lbp_test = lbp_val_norm.values
x_fd_train = fd_train_norm.values
x_fd_val = fd_val_norm.values
x_color_som_train = color_som_train_norm.values
x_color_som_val = color_som_val_norm.values
'''

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
#X_train = x_train
#X_val = x_test
Y_train = y_train
Y_val = y_test

# be sure of input layer!
lbp_model = Sequential()
lbp_model.add(Dense(64, activation='relu', input_shape=(lbp_input_size,)))
#lbp_model.add(Dropout(0.5))
lbp_model.add(Dense(64, activation='relu'))
#lbp_model.add(Dropout(0.5))
lbp_model.add(Dense(64, activation='relu'))
#lbp_model.add(Dropout(0.5))
lbp_model.add(Dense(64, activation='relu'))
#lbp_model.add(Dropout(0.5))
lbp_model.add(Dense(64, activation='relu'))
#lbp_model.add(Dropout(0.5))
lbp_model.add(Dense(num_classes, activation='softmax'))

fd_model = Sequential()
fd_model.add(Dense(512, activation='relu', input_shape=(fd_input_size,)))
fd_model.add(Dropout(0.5))
fd_model.add(Dense(512, activation='relu'))
fd_model.add(Dropout(0.5))
fd_model.add(Dense(512, activation='relu'))
fd_model.add(Dropout(0.5))
fd_model.add(Dense(512, activation='relu'))
fd_model.add(Dropout(0.5))
fd_model.add(Dense(512, activation='relu'))
fd_model.add(Dropout(0.5))
fd_model.add(Dense(num_classes, activation='softmax'))

color_som_model = Sequential()
color_som_model.add(Dense(512, activation='relu', input_shape=(color_som_input_size,)))
color_som_model.add(Dropout(0.5))
color_som_model.add(Dense(512, activation='relu'))
color_som_model.add(Dropout(0.5))
color_som_model.add(Dense(512, activation='relu'))
color_som_model.add(Dropout(0.5))
color_som_model.add(Dense(512, activation='relu'))
color_som_model.add(Dropout(0.5))
color_som_model.add(Dense(512, activation='relu'))
color_som_model.add(Dropout(0.5))
color_som_model.add(Dense(num_classes, activation='softmax'))

merged_output = add([lbp_model.output, fd_model.output, color_som_model.output])

model = Model([lbp_model.input, fd_model.input, color_som_model.input], merged_output)

model.summary()

model.compile(loss='categorical_crossentropy',
    optimizer=Adam(lr=learning_rate),
    metrics=['accuracy'])


keras.utils.plot_model(model, to_file='test_keras_plot_model.png', show_shapes=True)
IPython.display.Image("test_keras_plot_model.png")


history = model.fit([lbp_train_norm, fd_train_norm, color_som_train_norm], Y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=2,
    validation_data=([lbp_val_norm, fd_val_norm, color_som_val_norm], Y_val),
    callbacks=[early_stopping])

# plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
#plt.show()
plt.savefig('accuracy.svg', format='svg')

plt.clf()

# plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
#plt.show()
plt.savefig('loss.svg', format='svg')

print('test before save')
score = model.evaluate([lbp_val_norm, fd_val_norm, color_som_val_norm], y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
 
