#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 11:48:09 2021

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

    temp = np.reshape(np.transpose(tensor, new_indices), (tensor.shape[mode], -1), order='F')
    return temp.conj()


def construct_Y_given_E(E, D, F, G, A):
    """
        This function constructs y_hat = A * (kronecker[D, E, f_k]) * vec(G). Normally,
        this would be of dimensions (m x 1), but we construct this for
        each data sample, and thus is of dimensions (mq x 1).
        
        Arguments:
            E: Factor matrix of dimensions (n_2 x r_2)
            D: Factor matrix of dimensions (n_1 x r_1)
            F: Factor matrix of dimensions (q x r_3)
            G: Core tensor of dimensions (r_1, r_2, r_3)
            A: Sampling vectors of dimensions (n x m x q).
    """
        
    n = A.shape[0]
    m = A.shape[1]
    q = A.shape[2]

    n_1 = D.shape[0]
    n_2 = int(n / n_1)
    
    r_2 = G.shape[1] # rank
    
    E_transpose = np.reshape(E, (r_2, n_2), order='F')
    E=E_transpose.T
    
    vec_G = np.reshape(G, (-1, ), order='F')

    Y_all = np.zeros((m*q, ), dtype=np.complex)
    
    st = 0
    en = m

    for k in range(q):
        f_k_row = np.reshape(F[k], (1, -1), order='F')
        vec_x_k = tl.tenalg.kronecker([f_k_row, E, D]) @ vec_G
        y_k = A[:, :, k].conj().T @ vec_x_k
        Y_all[st:en] = y_k.squeeze()
    
        st += m
        en += m
        
        
    return Y_all
        
        
def constructSolutions(C_y, D, F, G, A):
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
    
    n_1 = D.shape[0]
    n_2 = int(n / n_1)
    
    r_2 = G.shape[1] # rank

    solved_E = np.zeros((n_2*r_2, ), dtype=np.complex)
    
    st = 0
    en = m

    for k in range(q):
        f_k_row = np.reshape(F[k], (1, -1))
        U = unfold(G, 1) @ np.kron(f_k_row, D).conj().T
        U_kron = np.kron(np.identity(n_2), U.conj().T)
        #U_reshape = np.reshape(U_kron, (n_1, n_2, -1), order='F')
        #new_U = np.reshape(U_reshape, (n_1*n_2, -1))
        
        #V = A[:, :, k].conj().T @ new_U
        V = A[:, :, k].conj().T @ U_kron
        # maybe edit here
        V_y = V.conj().T @ C_y[st:en]

        solved_E += V_y
        
        st += m
        en += m 

    return solved_E


def tucker_cgls_E(D, F, G, A_sample, C_y, max_iter=50, tol=1e-6):

    # initializing
    r = C_y
    s = constructSolutions(C_y=C_y, D=D, F=F, G=G, A=A_sample)
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
        
        q = construct_Y_given_E(E=p, D=D, F=F, G=G, A=A_sample)
        
        delta = np.linalg.norm(q)**2
        
        alpha = gamma / delta
        
        # make updates
        x = x + alpha*p
        r = r - alpha*q
        
        # update s
        s = constructSolutions(C_y=r, D=D, F=F, G=G, A=A_sample)
        
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