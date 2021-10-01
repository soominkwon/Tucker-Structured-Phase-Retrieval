#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 12:42:49 2021

@author: soominkwon
"""

import numpy as np
import tensorly as tl

def unfold(tensor, mode):
    """Simple unfolding function
        Moves the `mode` axis to the beginning and reshapes in Fortran order
    """
    n_dim = tensor.ndim
    indices = np.arange(n_dim).tolist()    
    element = indices.pop(mode)
    new_indices = ([element] + indices)  

    return np.reshape(np.transpose(tensor, new_indices).conj(), (tensor.shape[mode], -1), order='F')


def higherOrderSVD(tucker_rank, true_X):
    """
        Spectral initialization technique for Tucker decomposition.
        
    """
    
    # creating factor matrix A
    unfold_X_1 = unfold(true_X, 0)
    U, S, Vh = np.linalg.svd(unfold_X_1 , full_matrices=True)
    
    A = U[:, :tucker_rank[0]]
    
    # creating factor matrix B
    unfold_X_2 = unfold(true_X, 1)
    U, S, Vh = np.linalg.svd(unfold_X_2 , full_matrices=True)
    
    B = U[:, :tucker_rank[1]]

    # creating factor matrix C
    unfold_X_3 = unfold(true_X, 2)
    U, S, Vh = np.linalg.svd(unfold_X_3 , full_matrices=True)
    
    C = U[:, :tucker_rank[2]] 
    
    # creating core tensor D
    D = tl.tenalg.multi_mode_dot(true_X, [A.conj().T, B.conj().T, C.conj().T])
    
    dictionary = {}
    
    dictionary[0] = A
    dictionary[1] = B
    dictionary[2] = C
    dictionary[3] = D
    
    return dictionary