import numpy as np

def estimate(particles, particles_w):
    estimate = np.zeros(particles.shape[1])
    
    for i, particle in enumerate(particles):
        estimate += particle * particles_w[i]
        
    estimate = estimate / np.sum(particles_w)

    return estimate
        
    
    