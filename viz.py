# Python code with OpenCV and Numpy to load files from .npy array and save in a form of tiff's 

import cv2
import numpy as np

name = 'averaged_pos_'
cpos =  np.load(name+'control.npy');
spos = np.load(name+'shiz.npy');

%mkdir averaged2
%cd averaged2
cpos = 255*cpos/cpos.max()
spos = 255*spos/spos.max()

for i in range(0, cpos.shape[2]):  
    cv2.imwrite('s_C001Z%03d'%i+'.tif',  cpos[:,:,i])
    cv2.imwrite('s_C002Z%03d'%i+'.tif',  spos[:,:,i])