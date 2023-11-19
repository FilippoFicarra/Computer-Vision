import time
import numpy as np

# run `pip install scikit-image` to install skimage, if you haven't done so
from skimage import io, color
from skimage.transform import rescale

def distance(x, X):
    return np.sqrt(np.sum((x - X) ** 2, axis=1))

def gaussian(dist, bandwidth):
    K = np.exp(-(dist ** 2) / (2 * bandwidth ** 2))
    return K
    
def update_point(weight, X):
    num = np.sum(weight.reshape(-1, 1) * X, axis=0)
    den = np.sum(weight)
    return num / den

def meanshift_step(X, bandwidth=1):
    y = np.copy(X)
    for i in range(X.shape[0]):
        dist = distance(y[i], X)
        weight = gaussian(dist, bandwidth)
        y[i] = update_point(weight, X)
    return y

def meanshift(X):
    print(f"Running meanshift with bandwidth {bandwidth}")
    for _ in range(20):
        X = meanshift_step(X, bandwidth=bandwidth)
    return X

scale = 0.5    # downscale the image to run faster

# Load image and convert it to CIELAB space
image = rescale(io.imread('eth.jpg'), scale, channel_axis=-1)
image_lab = color.rgb2lab(image)
shape = image_lab.shape # record image shape
image_lab = image_lab.reshape([-1, 3])  # flatten the image

for bandwidth in [1,2,3,4,5,6,7]:
    # Run your mean-shift algorithm
    try:
        t = time.time()
        X = meanshift(image_lab)
        t = time.time() - t
        print ('Elapsed time for mean-shift: {}'.format(t))

        # Load label colors and draw labels as an image
        colors = np.load('colors.npz')['colors']
        colors[colors > 1.0] = 1
        colors[colors < 0.0] = 0

        centroids, labels = np.unique((X / 4).round(), return_inverse=True, axis=0)
        print(f"Number of clusters: {len(centroids)}")

        result_image = colors[labels].reshape(shape)
        result_image = rescale(result_image, 1 / scale, order=0, channel_axis=-1)     # resize result image to original resolution
        result_image = (result_image * 255).astype(np.uint8)
        io.imsave(f'result_{bandwidth}.png', result_image)
    except:
        print(f"bandwidth {bandwidth} failed")
