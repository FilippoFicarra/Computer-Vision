import numpy as np
from scipy.spatial.distance import cdist

def ssd(desc1, desc2):
    '''
    Sum of squared differences
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - distances:    - (q1, q2) numpy array storing the squared distance
    '''
    assert desc1.shape[1] == desc2.shape[1]
    # TODO: implement this function please
    return cdist(desc1, desc2, 'sqeuclidean')

def match_descriptors(desc1, desc2, method = "one_way", ratio_thresh=0.5):
    '''
    Match descriptors
    Inputs:
    - desc1:        - (q1, feature_dim) descriptor for the first image
    - desc2:        - (q2, feature_dim) descriptor for the first image
    Returns:
    - matches:      - (m x 2) numpy array storing the indices of the matches
    '''
    assert desc1.shape[1] == desc2.shape[1]
    distances = ssd(desc1, desc2)
    q1, q2 = desc1.shape[0], desc2.shape[0]
    matches = None
    if method == "one_way": # Query the nearest neighbor for each keypoint in image 1
        # TODO: implement the one-way nearest neighbor matching here
        # You may refer to np.argmin to find the index of the minimum over any axis
        matches = np.argmin(distances, axis=1)
        matches = np.array([(i, match) for i, match in enumerate(matches)])
    elif method == "mutual":
        # TODO: implement the mutual nearest neighbor matching here
        # You may refer to np.min to find the minimum over any axis
        matches_1 = np.argmin(distances, axis=1)
        matches_1 = np.array([(i, match) for i, match in enumerate(matches_1)])

        matches_2 = np.argmin(distances, axis=0)
        matches_2 = np.array([(i, match) for i, match in enumerate(matches_2)])

        matches_2 = matches_2[:, [1, 0]]

        matches_1_set = set([tuple((match[0], match[1])) for match in matches_1])
        matches_2_set = set([tuple((match[0], match[1])) for match in matches_2])

        matches = list(matches_1_set.intersection(matches_2_set))
        matches = np.array([[match[0], match[1]] for match in matches])

    elif method == "ratio":
        # TODO: implement the ratio test matching here
        # You may use np.partition(distances,2,axis=0)[:,1] to find the second smallest value over a row
        matches = np.argmin(distances, axis=1)
        matches = np.array([(i, match) for i, match in enumerate(matches)])

        partition_first= np.partition(distances,1,axis=1)[:, 0]

        partition_second = np.partition(distances,1,axis=1)[:, 1]

        matches = matches[partition_first/partition_second < ratio_thresh ]
    else:
        matches = np.array([[]])
    return matches

