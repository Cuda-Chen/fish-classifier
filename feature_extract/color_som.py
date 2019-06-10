from skimage import io
from minisom import MiniSom
import numpy as np

# read input image
img = io.imread("../data/003_0.bmp")

# reshape the pixels matrix and nomalize it
# (height, width, channel) -> (height * width, channel)
pixels = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2])) / 255

# SOM initialization and training
print("training...")
som = MiniSom(4, 4, img.shape[2], sigma=1.0,
             learning_rate=0.1, neighborhood_function='gaussian') # 4x4 = 16 final colors
#som.random_weights_init(pixels)

# initialize SOM's wieghts with PCA technique (to span the first two principal components)
# it seems that using this gets stable result
som.pca_weights_init(pixels)
starting_weights = som.get_weights().copy()  # saving the starting weights

# train SOM by picking samples at random from data
# and set maximum iteration to 500
som.train_random(pixels, 500, verbose=True)

print('quantization...')
qnt = som.quantization(pixels)  # quantize each pixels of the image
print('building new image...')
clustered = np.zeros(img.shape)
for i, q in enumerate(qnt):  # place the quantized values into a new image
    clustered[np.unravel_index(i, dims=(img.shape[0], img.shape[1]))] = q
print('done.')

# print starting and learned weights
# each row of the list represents the pixel value (color) of certain element
ending_weights = som.get_weights()
print("before learning")
print(starting_weights)
print("\nafter learning")
print(ending_weights)

# output learned color is csv
a = np.asarray(ending_weights)
a = a.reshape(1, 3 * 4 * 4)
np.savetxt("foo.csv", a, delimiter=",", fmt='%1.6f')
