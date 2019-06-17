#!/usr/bin/python
# -*- coding: UTF-8 -*-

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
import sys 
import numpy as np
import pandas as pd

# 41 + 1 classes
cls_list = ['10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
    '1', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
    '2', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39',
    '3', '40', '41', '42', '4', '5', '6', '7', '8', '9']
class_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 
    '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
    '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
    '31', '32', '33', '34', '35', '36', '37', '38', '39', '40',
    '41', '42']
'''
lbp_test = pd.read_csv('../LBP_feature_val.csv')
fd_test = pd.read_csv('../fd_feature_val.csv')
color_som_test = pd.read_csv('../color_som_val.csv')
'''
lbp_test = pd.read_csv('../LBP_20190604.csv')
fd_test = pd.read_csv('../fd_20190604.csv')
color_som_test = pd.read_csv('../color_som_20190604.csv')
test_label = pd.read_csv('../val_label.csv')

x_test = pd.concat([lbp_test, fd_test, color_som_test], axis=1).values
test_label_list = test_label['class_no'].tolist()
onehot_test = []
for value in test_label_list:
    class_id = [0 for _ in range(len(class_list))]
    class_id[value - 1] = 1
    onehot_test.append(class_id)
y_test = pd.DataFrame(onehot_test)

net = load_model('my_dnn.h5')
'''
score = net.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
'''
'''
pred = net.predict(x_test)[0]
top_inds = pred.argsort()[::-1][:5]
for i in top_inds:
    print('    {:.3f}  {}'.format(pred[i], cls_list[i]))
'''
#x_test = x_test.transpose()
print(x_test.shape)
for index in range(x_test.shape[0]):
    x = np.expand_dims(x_test[index][:], axis=0)
    pred = net.predict(x)[0]
    top_inds = pred.argsort()[::-1][:5]
    for i in top_inds:
        print(i)
        print('    {:.3f}  {}'.format(pred[i], cls_list[i]))
    print('=====')
