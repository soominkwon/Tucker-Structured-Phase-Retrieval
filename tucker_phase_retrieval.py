#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 12:31:13 2021

@author: soominkwon
"""

import numpy as np
import tensorly as tl
from tucker_cgls_D import tucker_cgls_D
from tucker_cgls_E import tucker_cgls_E
from tucker_cgls_core import tucker_cgls_G
from higher_order_SVD import higherOrderSVD
from reshaped_wirtinger_flow import rwf_fit
from reconstruct_tensor import reconstructTucker
import matplotlib.pyplot as plt

def unfold(tensor, mode):
    """Simple unfolding function
        Moves the `mode` axis to the beginning and reshapes in Fortran order
    """
    n_dim = tensor.ndim
    indices = np.arange(n_dim).tolist()    
    element = indices.pop(mode)
    new_indices = ([element] + indices)  

    temp = np.reshape(np.transpose(tensor, new_indices), (tensor.shape[mode], -1), order='F')
    return temp.conj()


def spectralInit(tucker_rank, image_dims, A, Y):
    """ Function for spectral initialization with higher-order SVD.
    
        Arguments:
            tucker_rank: List of mulitilinear ranks
            image_dims: Tuple of dimensions of the image
            Y: Observation matrix with dimensions (m x q)
            A: Measurement tensor with dimensions (n x m x q)

    """    
    
    # initializing
    m = Y.shape[0]
    q = Y.shape[1] 
    n_1 = image_dims[0]
    n_2 = image_dims[1]

    Y_tensor = np.zeros((n_1, n_2, q), dtype=np.complex)
    
    # looping through each frame
    for k in range(q):
        y_k = Y[:, k]
        trunc_val = 9*y_k.mean()
        trunc_y_k = np.where(np.abs(y_k)<=trunc_val, y_k, 0)
        Y_u = A[:, :, k] @ np.diag(trunc_y_k) @ A[:, :, k].conj().T
        Y_u = (1/m)*Y_u
        U, S, V = np.linalg.svd(Y_u)
        vec_x_init = U[:, 0]
        x_init = np.reshape(vec_x_init, (n_1, n_2), order='F')
        Y_tensor[:, :, k] = x_init
    
    factor_mats = higherOrderSVD(tucker_rank=tucker_rank, true_X=Y_tensor)
    
    return factor_mats


def updateC(A, D, E, F, G):
    """ Function to update the diagonal phase matrix C.
    
        Arguments: 
            A: Measurement tensor with dimensions(n x m x q)
            D: Factor matrix of dimensions (n_1 x r)
            E: Factor matrix of dimensions (n_2 x r)
            F: Factor matrix of dimensions (q x r)
            
        Returns:
            C_tensor: Tensor where the frontal slices represent C_k (diagonal phase matrix)
                        with dimensions (m x m x q)
    """
    
    #n = A.shape[0]
    m = A.shape[1]    
    q = A.shape[2]
    
    vec_G = np.reshape(G, (-1, ), order='F')
    
    C_tensor = np.zeros((m, m, q), dtype=np.complex)
    
    for k in range(q):
        A_k = A[:, :, k]
        #f_k_row = np.reshape(F[k], (1, -1), order='F')
        
        vec_x_hat = tl.tenalg.kronecker([F[k], E, D]) @ vec_G
        y_hat = A_k.conj().T @ vec_x_hat
        
        #phase_y = np.sign(y_hat)
        phase_y = np.exp(1j*np.angle(y_hat))
        C_k = np.diag(phase_y)
        C_tensor[:, :, k] = C_k
               
    return C_tensor


def tucker_fit(tucker_rank, image_dims, A, Y, max_iters):
    
    # initializing factors
    cp_factors = spectralInit(tucker_rank=tucker_rank, image_dims=image_dims, A=A, Y=Y)
    print('Spectral Initialization for Tucker Decomposition Complete.')
    
    D = cp_factors[0]
    E = cp_factors[1]
    F = cp_factors[2]
    G = cp_factors[3]
    
    F = np.zeros(F.shape, dtype=np.complex)
        
    n_1 = image_dims[0]
    n_2 = image_dims[1]
    m = A.shape[1]
    q = A.shape[2]
        
    # starting main loop
    Ysqrt = np.sqrt(Y)

    for i in range(max_iters):
        print('Current Iteration:', i)

        
        # solving a RWF problem for each f_k
        for k in range(q):
            y_k = Ysqrt[:, k]
            A_k = A[:, :, k]   
            #C_y = C_all[:, :, k] @ Ysqrt[:, k]  
            A_k = A[:, :, k]
            
            H = unfold(G, 2) @ np.kron(E, D).conj().T
            rwf_A = A_k.conj().T @ H.conj().T
            f_k = rwf_fit(y=y_k, A=rwf_A.conj().T)
            #J = A_k.conj().T @ H.T
            #inv_mat = np.linalg.inv(J.conj().T @ J)
            #f_k = inv_mat @ J.conj().T @ C_y
            
            F[k] = f_k
            print('Update F', k, 'Complete.')
        # update D
        st = 0
        en = m
    
        # update phase matrix
        C_all = updateC(A=A, D=D, E=E, F=F, G=G)
        C_y_vec = np.zeros((m*q, ), dtype=np.complex)

            
        for k in range(q):
            C_y = C_all[:, :, k] @ Ysqrt[:, k]
            C_y_vec[st:en] = C_y
            
            st += m
            en += m


        D_vec = tucker_cgls_D(E=E, F=F, G=G, A_sample=A, C_y=C_y_vec)
        D = np.reshape(D_vec, (n_1, tucker_rank[0]), order='F')
        print('Update D Completed.')

        E_vec = tucker_cgls_E(D=D, F=F, G=G, A_sample=A, C_y=C_y_vec)
        E_transpose = np.reshape(E_vec, (tucker_rank[1], n_2), order='F')
        E = E_transpose.conj().T
        print('Update E Completed.')
        
        G_vec = tucker_cgls_G(D=D, E=E, F=F, A_sample=A, C_y=C_y_vec)
        G = np.reshape(G_vec, (tucker_rank[0], tucker_rank[1], tucker_rank[2]), order='F')
        print('Update G Completed.')
        
    X_tucker = reconstructTucker(D=D, E=E, F=F, G=G)
    return X_tucker   
    
    
    
    