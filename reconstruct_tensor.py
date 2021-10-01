#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 12:24:59 2021

@author: soominkwon
"""

import numpy as np
import tensorly as tl


def reconstructTucker(D, E, F, G):
    """
        Reconstructs factor matrices and core tensor back into its tensor form.
    """
    
    n_1 = D.shape[0]
    n_2 = E.shape[0]
    n_3 = F.shape[0]
    
    X = np.zeros((n_1, n_2, n_3), dtype=np.complex)
    vec_G = np.reshape(G, (-1, ), order='F')
    
    for k in range(n_3):
        f_k_row = np.reshape(F[k], (1, -1), order='F')
        
        vec_X = tl.tenalg.kronecker([f_k_row, E, D]) @ vec_G
        X_slice = np.reshape(vec_X, (n_1, n_2), order='F')
        
        X[:, :, k] = X_slice
        
    return X


