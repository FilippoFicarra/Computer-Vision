import numpy as np
def resample(particles, particles_w):
    
    index = np.random.choice(particles.shape[0], particles.shape[0], p=particles_w)
    return particles[index], particles_w[index]