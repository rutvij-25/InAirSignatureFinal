from dtaidistance.dtw_ndim import distance_fast as df2
import numpy as np
from scipy.fft import fft2,ifft2
import matplotlib.pyplot as plt
from scipy import spatial
from PIL import Image
import os

import numpy as np

def dtw_distance(s1, s2):
    """
    Compute the Dynamic Time Warping (DTW) distance between two 2D sequences.

    Arguments:
    s1, s2 -- Input sequences (numpy arrays)

    Returns:
    dtw -- The DTW distance between the two sequences
    """
    len_s1 = s1.shape[0]
    len_s2 = s2.shape[0]

    # Create a cost matrix with infinite values
    cost_matrix = np.zeros((len_s1, len_s2)) + np.inf

    # Initialize the first row and column of the cost matrix
    cost_matrix[0, 0] = 0

    # Fill in the cost matrix
    for i in range(1, len_s1):
        for j in range(1, len_s2):
            cost = np.linalg.norm(s1[i] - s2[j])  # Euclidean distance between two vectors
            cost_matrix[i, j] = cost + min(cost_matrix[i-1, j], cost_matrix[i, j-1], cost_matrix[i-1, j-1])

    # The DTW distance is the value in the bottom right cell of the cost matrix
    dtw = cost_matrix[len_s1-1, len_s2-1]
    
    return dtw


def read_database(NAME):
    images = []
    strokes = [] 
    for i in os.listdir(f'Database/{NAME}'):
        if(i=='images'):
            for j in os.listdir(f'Database/{NAME}/images'):
                images.append(np.asarray(Image.open(f'Database/{NAME}/images/'+j))/255)
        if(i=='strokes'):
            for j in os.listdir(f'Database/{NAME}/strokes'):
                strokes.append(np.load(f'Database/{NAME}/strokes/'+j))   
    return images,strokes

def mtdtw_matching(strokelist,strokeref):
    m = []
    for i in strokelist:
        dist = dtw_distance(i,strokeref)
        m.append(dist)
    
    return np.mean(np.array(m))

def fft_matching(imglist,imgref):
    m = []
    for i in imglist:
        cdist = 1 - spatial.distance.cosine(fft2(i).reshape(-1), fft2(imgref).reshape(-1))
        m.append(cdist)
    
    return np.mean(np.array(m))

def moving_avg(traj,N):
    xtraj = traj[:,0]
    ytraj = traj[:,1]
    xmean = [np.mean(xtraj[x:x+N]) for x in range(len(xtraj)-N+1)]
    ymean = [np.mean(ytraj[x:x+N]) for x in range(len(ytraj)-N+1)]
    newtraj = np.array([[i,j] for i,j in zip(xmean,ymean)])
    return newtraj
    
def match(NAME,imgref,strokeref):
    images,strokes = read_database(NAME)
    mtdtwscore = mtdtw_matching(strokes, strokeref)
    print(mtdtwscore)
    fftscore = fft_matching(images, imgref)
    if mtdtwscore<1.5 and fftscore>0.98:
        return True
    else:
        return False