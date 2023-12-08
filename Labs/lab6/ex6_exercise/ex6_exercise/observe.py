import numpy as np
from color_histogram import color_histogram
from chi2_cost import chi2_cost

def observe(particles, frame, bbox_height, bbox_width, hist_bin, hist, sigma_observe):
    
    particles_w = np.zeros(particles.shape[0])
    for i, particle in enumerate(particles):
        x_center = particle[0]
        y_center = particle[1]
        
        x_min = np.clip(int(x_center - bbox_width/2), 0, frame.shape[1]-1)
        y_min = np.clip(int(y_center - bbox_height/2), 0, frame.shape[0]-1)
        x_max = np.clip(int(x_center + bbox_width/2), 0, frame.shape[1]-1)
        y_max = np.clip(int(y_center + bbox_height/2), 0, frame.shape[0]-1)
        
        histogram = color_histogram(x_min, y_min, x_max, y_max, frame, hist_bin)
        
        chi_2 = chi2_cost(histogram, hist)
        
        particles_w[i] = 1/(np.sqrt(2*np.pi)*sigma_observe) * np.exp(-chi_2**2/(2*sigma_observe**2))
        
    particles_w = particles_w / np.sum(particles_w)
    
    return particles_w