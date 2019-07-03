from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
import sys
import numpy as np
from fishconfig import config
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

# 41 classes
classList = ['10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
    '1', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
    '2', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39',
    '3', '40', '41', '4', '5', '6', '7', '8', '9']

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

def get_metrics(y_test, y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted, average='weighted')
    recall = recall_score(y_test, y_predicted, average='weighted')
    f1 = f1_score(y_test, y_predicted, average='weighted')
    return accuracy, precision, recall, f1


DATASET_PATH = config.DATASET_PATH
IMAGE_SIZE = config.IMAGE_SIZE
BATCH_SIZE = config.BATCH_SIZE

#files = sys.argv[1:]

net = load_model('model-resnet50-final.h5')


#net = load_model('my_dnn.h5')

test_datagen = ImageDataGenerator()
test_batches = test_datagen.flow_from_directory(DATASET_PATH + '/val',
                                                target_size=IMAGE_SIZE,
                                                interpolation='bicubic',
                                                class_mode='categorical',
                                                shuffle=False)
# output the index number of each class
for cls, idx in test_batches.class_indices.items():
    print('Class #{} = {}'.format(idx, cls))

print('number of samples:', test_batches.samples)

score = net.evaluate_generator(test_batches, steps=test_batches.samples, verbose=0)
print('Test loss:', score[0])
print('Test top-1 accuracy:', score[1])
print('Test top-5 accuracy:', score[2])

Y_pred = net.predict_generator(test_batches, steps=test_batches.samples)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_batches.classes, y_pred))
ax = plot_confusion_matrix(test_batches.classes, y_pred, classList)
plt.show()
plt.savefig('test.png')
print('Classification Report')
target_names = classList
print(classification_report(test_batches.classes, y_pred, target_names=target_names))
'''
# make prediction
for f in files:
    img = image.load_img(f, target_size=IMAGE_SIZE)
    if img is None:
        continue
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    pred = net.predict(x)[0]
    top_inds = pred.argsort()[::-1][:5]
    print(f)
    for i in top_inds:
        print('    {:.3f}  {}'.format(pred[i], cls_list[i]))
'''
