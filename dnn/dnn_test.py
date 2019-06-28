#!/usr/bin/python
# -*- coding: UTF-8 -*-

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
import sys 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

# 41 classes
cls_list = ['10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
    '1', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
    '2', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39',
    '3', '40', '41', '4', '5', '6', '7', '8', '9']
class_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 
    '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
    '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
    '31', '32', '33', '34', '35', '36', '37', '38', '39', '40',
    '41']
classList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41]

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    classes = classList
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

lbp_test = pd.read_csv('../LBP_feature_val.csv')
fd_test = pd.read_csv('../fd_feature_val.csv')
color_som_test = pd.read_csv('../color_som_val.csv')
'''
lbp_test = pd.read_csv('../LBP_20190604.csv')
fd_test = pd.read_csv('../fd_20190604.csv')
color_som_test = pd.read_csv('../color_som_20190604.csv')
'''
test_label = pd.read_csv('../val_label.csv')

x_test = pd.concat([lbp_test, fd_test, color_som_test], axis=1).values
test_label_list = test_label['class_no'].tolist()
test_label_nparr = test_label['class_no'].values
print(test_label_nparr.shape)
onehot_test = []
for value in test_label_list:
    class_id = [0 for _ in range(len(class_list))]
    class_id[value - 1] = 1
    onehot_test.append(class_id)
y_test = pd.DataFrame(onehot_test)

net = load_model('my_dnn.h5')

score = net.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
'''
y_new = net.predict(x_test)
np.set_printoptions(suppress=True)
for i in range(y_new.shape[0]):
    if ((np.argmax(y_new[i]) + 1) == test_label_nparr[i]):
        print("Predicted=%s, predict class=%s, truth=%s, index=%s" % (y_new[i], np.argmax(y_new[i]) + 1, test_label_nparr[i], i + 1))
'''

y_pred_class = net.predict_classes(x_test)
y_pred_class = y_pred_class + 1
print(test_label_nparr.shape)
print(y_pred_class.shape)
#print(y_pred_class.shape)
from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(test_label_nparr, y_pred_class, labels=classList)

plot_confusion_matrix(test_label_nparr, y_pred_class, classes=classList,
                      title='Confusion matrix, without normalization')
plt.show()
'''
pred = net.predict(x_test)[0]
top_inds = pred.argsort()[::-1][:5]
for i in top_inds:
    print('    {:.3f}  {}'.format(pred[i], cls_list[i]))
'''
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
'''
