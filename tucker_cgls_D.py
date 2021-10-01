#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 18:38:27 2021

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

    temp = np.reshape(np.transpose(tensor, new_indices).conj(), (tensor.shape[mode], -1), order='F')
    return temp


def construct_Y_given_D(D, E, F, G, A):
    """
        This function constructs y_hat = A * (kronecker[D, E, f_k]) * vec(G). Normally,
        this would be of dimensions (m x 1), but we construct this for
        each data sample, and thus is of dimensions (mq x 1).
        
        Arguments:
            D: Factor matrix of dimensions (n_1 x r_1)
            E: Factor matrix of dimensions (n_2 x r_2)
            F: Factor matrix of dimensions (q x r_3)
            G: Core tensor of dimensions (r_1, r_2, r_3)
            A: Sampling vectors of dimensions (n x m x q).
    """
        
    n = A.shape[0]
    m = A.shape[1]
    q = A.shape[2]

    n_2 = E.shape[0]
    n_1 = int(n / n_2)
    
    r_1 = G.shape[0] # rank
    
    D = np.reshape(D, (n_1, r_1), order='F')
    vec_G = np.reshape(G, (-1, ), order='F')
    
    Y_all = np.zeros((m*q, ), dtype=np.complex)
    
    st = 0
    en = m

    for k in range(q):
        f_k_row = np.reshape(F[k], (1, -1), order='F')
        vec_x_k = tl.tenalg.kronecker([f_k_row, E, D]) @ vec_G
        y_k = A[:, :, k].conj().T @ vec_x_k
        Y_all[st:en] = y_k.squeeze()
    
        st+=m
        en+=m
        
        
    return Y_all
        
        
def constructSolutions(C_y, E, F, G, A):
    """
           
        Arguments:
            C_y: C * \sqrt{y} for all q (m*q x 1)
            E: Factor matrix of dimensions (n_2 x r)
            F: Factor matrix of dimensions (q x r)
            G: Core tensor of dimensions (r_1, r_2, r_3)
            A: Sampling vectors of dimensions (n x m x q).
    """      
    
    n = A.shape[0]
    m = A.shape[1]
    q = A.shape[2]
    
    n_2 = E.shape[0]
    n_1 = int(n / n_2)
    
    r_1 = G.shape[0] # rank

    solved_D = np.zeros((n_1*r_1, ), dtype=np.complex)
    
    st = 0
    en = m

    for k in range(q):
        f_k_row = np.reshape(F[k], (1, -1), order='F')
        S = unfold(G, 0) @ np.kron(f_k_row, E).conj().T
        S_kron = np.kron(S.conj().T, np.identity(n_1))

        T_a = A[:, :, k].conj().T @ S_kron
        T_y = T_a.conj().T @ C_y[st:en]

        solved_D += T_y
        
        st += m
        en += m 

    return solved_D


def tucker_cgls_D(E, F, G, A_sample, C_y, max_iter=50, tol=1e-6):

    # initializing
    r = C_y
    s = constructSolutions(C_y=C_y, E=E, F=F, G=G, A=A_sample)
    n = s.shape[0]
    x = np.zeros((n, ), dtype=np.complex) # optimize variable

    # initializing for optimization
    p = s
    norms0 = np.linalg.norm(s)
    gamma = norms0**2
    normx = np.linalg.norm(x)**2
    xmax = normx
    
    iters = 0
    flag = 0
    
    while (iters < max_iter) and (flag == 0):
        #print('Current Iteration:', iters)
        
        q = construct_Y_given_D(D=p, E=E, F=F, G=G, A=A_sample)
        
        delta = np.linalg.norm(q)**2
        
        alpha = gamma / delta
        
        # make updates
        x = x + alpha*p
        r = r - alpha*q
        
        # update s
        s = constructSolutions(C_y=r, E=E, F=F, G=G, A=A_sample)
        
        norms = np.linalg.norm(s)
        gamma1 = gamma
        gamma = norms**2
        beta = gamma / gamma1
        
        # update p
        p = s + beta*p
        
        # convergence
        normx = np.linalg.norm(x)
        xmax = max(xmax, normx)
        flag = (norms <= norms0 * tol) or (normx * tol >= 1)
        
        iters += 1
    
    return x



        
        
        
        
        
        
        
        
        
        