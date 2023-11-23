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
    y = X.copy()
    for i in range(X.shape[0]):
        dist = distance(X[i], X)
        weight = gaussian(dist, bandwidth)
        y[i] = update_point(weight, X)
    return y

def meanshift(X):
    for _ in range(20):
        X = meanshift_step(X, bandwidth=bandwidth)
    return X

def shrink_labels(centroids, labels, colors):
    l , counts = np.unique(labels, return_counts=True)
    l = l[np.argsort(counts)[::-1]]
    
    big_clusters = l[:len(colors)]

    for i in range(len(l)):
        if labels[i] not in big_clusters:
            dist = distance(centroids[labels[i]], centroids[big_clusters])
            labels[i] = big_clusters[np.argmin(dist)]
    
    return labels % len(colors) 


scale = 0.5    # downscale the image to run faster

for bandwidth in [1, 2, 2.5, 3, 4, 5, 6, 7]:

    print(f"Running meanshift with bandwidth {bandwidth}")
    # Load image and convert it to CIELAB space
    image = rescale(io.imread('eth.jpg'), scale, channel_axis=-1)
    image_lab = color.rgb2lab(image)
    shape = image_lab.shape # record image shape
    image_lab = image_lab.reshape([-1, 3])  # flatten the image
    
    t = time.time()
    X = meanshift(image_lab)
    t = time.time() - t
    print ('Elapsed time for mean-shift: {}'.format(t))

    # Load label colors and draw labels as an image
    colors = np.load('colors.npz')['colors']
    colors[colors > 1.0] = 1
    colors[colors < 0.0] = 0

    centroids, labels = np.unique((X / 4).round(), return_inverse=True, axis=0)
    
    if len(centroids) > len(colors):
        print(f"Number of clusters ({len(centroids)}) is greater than number of colors ({len(colors)}).")
        labels = shrink_labels(centroids, labels, colors)
    
    print(f"Number of different labels: {len(np.unique(labels))}")

    result_image = colors[labels].reshape(shape)
    result_image = rescale(result_image, 1 / scale, order=0, channel_axis=-1)     # resize result image to original resolution
    result_image = (result_image * 255).astype(np.uint8)
    io.imsave(f'data/result_{bandwidth}.png', result_image)
    
