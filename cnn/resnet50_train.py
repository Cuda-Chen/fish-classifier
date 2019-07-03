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

# data agumentation

train_datagen = ImageDataGenerator(rotation_range=90,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   channel_shift_range=10,
                                   #validation_split=0.25,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='nearest')
'''
train_datagen = ImageDataGenerator()
'''
train_batches = train_datagen.flow_from_directory(DATASET_PATH + '/train',
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
for cls, idx in train_batches.class_indices.items():
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

print('train samples:', train_batches.samples)
print('validation samples:', valid_batches.samples)
#print('test samples:', test_batches.samples)

net_final.fit_generator(train_batches,
                        steps_per_epoch = train_batches.samples // BATCH_SIZE,
                        validation_data = valid_batches,
                        validation_steps = valid_batches.samples // BATCH_SIZE,
                        epochs = NUM_EPOCHS,
                        callbacks=[earlyStopping])

score = net_final.evaluate(valid_batches, steps=(valid_batches.samples / BATCH_SIZE), verbose=0)
print('Test loss:', score[0])
print('Test top-1 accuracy:', score[1])
print('Test top-5 accuracy:', score[2])

# evaluate!
# 41 classes
classList = ['10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
    '1', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
    '2', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39',
    '3', '40', '41', '4', '5', '6', '7', '8', '9']
f = '七星斑.jpg'
img = image.load_img(f, target_size=IMAGE_SIZE)
if img is None:
    print('input image not found')
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
pred = net_final.predict(x)[0]
top_inds = pred.argsort()[::-1][:5]
print(f)
print(pred.shape)
print(top_inds.shape)
for i in top_inds:
    print('    {:.3f}  {}'.format(pred[i], classList[i]))

net_final.save(WEIGHTS_FINAL)
#np.savetxt(str(args.seed), score, fmt='%1.4f')
