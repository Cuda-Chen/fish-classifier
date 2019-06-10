#!/usr/bin/python
# -*- coding: UTF-8 -*-

from __future__ import print_function

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

num_classes = 42 # 41 + 1
epochs = 100

# get your training and testing data here
# and to make sure to reshape and normalize!
# LBP currently
x_train
x_test

# convert class vectors to binary class matrices (one-hop encoding)
y_train
y_test

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(,)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
    optimizer=Adam(lr=1e-5),
    metrics=['accuracy'])

history = model.fit(x_train, y_train,
    epochs=epochs,
    verbose=1,
    validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
