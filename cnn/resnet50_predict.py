from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
import sys
import numpy as np

files = sys.argv[1:]

net = load_model('model-resnet50-final.h5')

# 41 + 1 classes
cls_list = ['10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
    '1', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
    '2', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39',
    '3', '40', '41', '42', '4', '5', '6', '7', '8', '9']

# make prediction
for f in files:
    img = image.load_img(f, target_size=(512, 512))
    if img is None:
        continue
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    pred = net.predict(x)[0]
    top_inds = pred.argsort()[::-1][:5]
    print(f)
    for i in top_inds:
        print('    {:.3f}  {}'.format(pred[i], cls_list[i]))
