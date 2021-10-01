#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 13:02:19 2021

@author: soominkwon
"""

import numpy as np
import matplotlib.pyplot as plt
from tucker_phase_retrieval import tucker_fit

# importing sample data
data_name = 'mouse_small_data.npz'

with np.load(data_name) as sample_data:
    vec_X = sample_data['arr_0']
    Y = sample_data['arr_1']
    A = sample_data['arr_2']
    
# initializing parameters
image_dims = [10, 30]
tucker_rank = [5, 10, 1]
iters = 5

# fitting new X
X_tucker = tucker_fit(tucker_rank=tucker_rank, image_dims=image_dims, 
                      A=A, Y=Y, max_iters=iters)

X = np.reshape(vec_X, (image_dims[0], image_dims[1], -1), order='F')

# plotting results
plt.imshow(np.abs(X[:, :, 0]), cmap='gray')
plt.title('True Image')
plt.show()

plt.imshow(np.abs(X_tucker[:, :, 0]), cmap='gray')
plt.title('Reconstructed Image via TS-LRPR')
plt.show()
    