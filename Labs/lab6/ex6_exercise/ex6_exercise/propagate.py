import numpy as np

def propagate(particles, frame_height, frame_width, params):
    delta_t = 1
    sigma_p = params["sigma_position"]
    sigma_v = params["sigma_velocity"]
    
    A = np.identity(particles.shape[1])
    model_noise_std = np.array([sigma_p, sigma_p])
    
    if params["model"] == 1:
        A[0, 2] = delta_t
        A[1, 3] = delta_t
        model_noise_std = np.array([sigma_p, sigma_p, sigma_v, sigma_v])

    w = np.random.randn(particles.shape[0], particles.shape[1]) * model_noise_std
    # np.random.normal(0, model_noise_std, particles.shape)
    
    
    particles = np.matmul(particles, A.T) + w
    
    
    particles[:, 0] = np.clip(particles[:, 0], 0, frame_width - 1)
    particles[:, 1] = np.clip(particles[:, 1], 0, frame_height - 1)
        
    return particles
    
    
    