from __future__ import print_function
import keras
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
#import tensorflow.keras.utils.layer_utils
from keras_contrib.applications.resnet import ResNet34, ResNet18
from fishconfig import config
import matplotlib.pyplot as plt
from numpy.random import seed
seed(1333)

def plot_keras_model(model, show_shapes=True, show_layer_names=True):
    from IPython.display import SVG
    from keras.utils.vis_utils import model_to_dot
    return SVG(model_to_dot(model, show_shapes=show_shapes, show_layer_names=show_layer_names).create(prog='dot', format='svg'))

DATASET_PATH = config.DATASET_PATH # dataset path
IMAGE_SIZE = config.IMAGE_SIZE # wrap the input image size
NUM_CLASSES = config.NUM_CLASSES
BATCH_SIZE = config.BATCH_SIZE
FREEZE_LAYERS = config.FREEZE_LAYERS
NUM_EPOCHS = config.NUM_EPOCHS
WEIGHTS_FINAL = 'model-resnet34-final.h5'

train_datagen = ImageDataGenerator()
train_batches = train_datagen.flow_from_directory(DATASET_PATH + '/train',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE)

valid_datagen = ImageDataGenerator()
valid_batches = valid_datagen.flow_from_directory(DATASET_PATH + '/val',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE)

# output the index number of each class
for cls, idx in train_batches.class_indices.items():
    print('Class #{} = {}'.format(idx, cls))

# use resnet34 and drop the last fully connected layer
net = ResNet18(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), 
    classes=NUM_CLASSES)

x = net.output
'''
x = Flatten()(x)

# add a dropout layer
x = Dropout(0.5)(x)

# add a dense layer with softmax
output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)
'''

# set the layers to be whether freezed or trained
#net_final = Model(inputs=net.input, outputs=output_layer)
net_final = Model(inputs=net.input, outputs=x)
'''
for layer in net_final.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in net_final.layers[FREEZE_LAYERS:]:
    layer.trainable = True
'''

net_final.compile(optimizer=Adam(lr=1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy'])

print(net_final.summary())

#temp = plot_keras_model(net_final, show_shapes=True, show_layer_names=False)

history = net_final.fit_generator(train_batches,
                        steps_per_epoch = train_batches.samples // BATCH_SIZE,
                        validation_data = valid_batches,
                        validation_steps = valid_batches.samples // BATCH_SIZE,
                        epochs = NUM_EPOCHS)

# plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.grid(linestyle='dotted')
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
plt.grid(linestyle='dotted')
#plt.show()
plt.savefig('loss.svg', format='svg')

print('test before save')
score = net_final.evaluate_generator(valid_batches, valid_batches.samples / BATCH_SIZE,
    verbose=0)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])
