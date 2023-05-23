import numpy as np
import cv2 as cv
import os
from utils import match,moving_avg
import matplotlib.pyplot as plt
from PIL import Image
from scipy import spatial
from scipy.fft import fft2,ifft2

imgref = np.asarray(Image.open('database/made/sign3.jpg'))/255
sign_new = np.load('database/made/sign3.npy')

m = match('aryan',imgref,sign_new)

if(m):
    print('Signature matched')
else:
    print('Signature did not match')