from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Flatten, Dense, Dropout
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.preprocessing import image
from fishconfig import config
from argparse import ArgumentParser
import numpy as np
from numpy.random import seed
import matplotlib.pyplot as plt
seed(1033)
'''
parser = ArgumentParser()
parser.add_argument('seed', type=int, help='seed')
args = parser.parse_args()

seed(args.seed)
'''
DATASET_PATH = config.DATASET_PATH # dataset path
IMAGE_SIZE = config.IMAGE_SIZE # wrap the input image size
NUM_CLASSES = config.NUM_CLASSES
BATCH_SIZE = config.BATCH_SIZE
FREEZE_LAYERS = config.FREEZE_LAYERS
NUM_EPOCHS = config.NUM_EPOCHS
WEIGHTS_FINAL = 'model-resnet50-final.h5'
earlyStopping = EarlyStopping(monitor='val_loss', patience=100)

train_datagen_ref = ImageDataGenerator()
train_batches_ref = train_datagen_ref.flow_from_directory(DATASET_PATH + '/train',
                                                          target_size=IMAGE_SIZE,
                                                          interpolation='bicubic',
                                                          class_mode='categorical',
                                                          shuffle=True,
                                                          batch_size=BATCH_SIZE)

#valid_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator()
valid_batches = valid_datagen.flow_from_directory(DATASET_PATH + '/val',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE)

# test dataset
'''
test_datagen = ImageDataGenerator()
test_batches = test_datagen.flow_from_directory(DATASET_PATH + '/val',
                                                target_size=IMAGE_SIZE,
                                                interpolation='bicubic',
                                                class_mode='categorical',
                                                shuffle=False,
                                                batch_size=BATCH_SIZE)
'''
# output the index number of each class
for cls, idx in train_batches_ref.class_indices.items():
    print('Class #{} = {}'.format(idx, cls))

# use pretrained resnet50 and drop the last fully connected layer
net = ResNet50(include_top=False, weights='imagenet', input_tensor=None,
               input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
x = net.output
x = Flatten()(x)

# add a dropout layer
x = Dropout(0.5)(x)

# add another Dense layer
#x = Dense(2048, activation='relu', name='readToOutput')(x)
#x = Dropout(0.5)(x)

# add a dense layer with softmax
output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)

# set the layers to be whether freezed or trained
net_final = Model(inputs=net.input, outputs=output_layer)
for layer in net_final.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in net_final.layers[FREEZE_LAYERS:]:
    layer.trainable = True

net_final.compile(optimizer=Adam(lr=1e-5),
                  loss='categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy'])

print(net_final.summary())

print('train_ref samples:', train_batches_ref.samples)
print('validation samples:', valid_batches.samples)
#print('test samples:', test_batches.samples)
print('train plain')
history_ref = net_final.fit_generator(train_batches_ref,
                        steps_per_epoch = train_batches_ref.samples // BATCH_SIZE,
                        validation_data = valid_batches,
                        validation_steps = valid_batches.samples // BATCH_SIZE,
                        epochs = NUM_EPOCHS,
                        callbacks=[earlyStopping])

plt.plot(history_ref.history['acc'])
plt.plot(history_ref.history['val_acc'])
plt.title('Top-1 accuracy curve without data augmentation')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.grid()
plt.savefig('top1_accuracy_ref.svg')

plt.clf()

plt.plot(history_ref.history['loss'])
plt.plot(history_ref.history['val_loss'])
plt.title('Loss curve without data augmentation')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.grid()
plt.savefig('loss_ref.svg')
