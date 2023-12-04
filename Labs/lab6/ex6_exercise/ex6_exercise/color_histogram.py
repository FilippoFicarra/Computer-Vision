import numpy as np

def color_histogram(xmin, ymin, xmax, ymax, frame, hist_bin):
    
    bounding_box = frame[ymin:ymax, xmin:xmax]
    hist_channels = []
    
    for channel in range(bounding_box.shape[2]):
        hist, _ = np.histogram(bounding_box[:,:,channel], bins=hist_bin)
        hist_channels.append(hist)
        
    histogram = np.concatenate(hist_channels)
    norm_histogram = histogram / np.sum(histogram)

    return norm_histogram