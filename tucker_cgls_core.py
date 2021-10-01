 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 12:19:11 2021

@author: soominkwon
"""

import numpy as np
import tensorly as tl


def construct_Y_given_G(G, D, E, F, A):
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
        
    m = A.shape[1]
    q = A.shape[2]
    
    vec_G = np.reshape(G, (-1, ), order='F')

    Y_all = np.zeros((m*q, ), dtype=np.complex)
    
    st = 0
    en = m

    for k in range(q):
        f_k_row = np.reshape(F[k], (1, -1))
        vec_x_k = tl.tenalg.kronecker([f_k_row, E, D]) @ vec_G
        y_k = A[:, :, k].conj().T @ vec_x_k
        Y_all[st:en] = y_k.squeeze()
    
        st+=m
        en+=m
             
    return Y_all
        
        
def constructSolutions(C_y, D, E, F, A):
    """
           
        Arguments:
            C_y: C * \sqrt{y} for all q (m*q x 1)
            D: Factor matrix of dimensions (n_1 x r_1)
            E: Factor matrix of dimensions (n_2 x r_2)
            F: Factor matrix of dimensions (q x r_3)
            A: Sampling vectors of dimensions (n x m x q).
    """      
    
    m = A.shape[1]
    q = A.shape[2]
    
    r_1 = D.shape[1]
    r_2 = E.shape[1]
    r_3 = F.shape[1]

    solved_G = np.zeros((r_1*r_2*r_3, ), dtype=np.complex)
    
    st = 0
    en = m

    for k in range(q):
        f_k_row = np.reshape(F[k], (1, -1), order='F')
        M = A[:, :, k].conj().T @ (tl.tenalg.kronecker([f_k_row, E, D]))
        # maybe edit here
        M_y = M.conj().T @ C_y[st:en]

        solved_G += M_y
        
        st += m
        en += m 

    return solved_G


def tucker_cgls_G(D, E, F, A_sample, C_y, max_iter=50, tol=1e-6):

    # initializing
    r = C_y
    s = constructSolutions(C_y=C_y, D=D, E=E, F=F, A=A_sample)
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
        
        q = construct_Y_given_G(G=p, D=D, E=E, F=F, A=A_sample)
        
        delta = np.linalg.norm(q)**2
        
        alpha = gamma / delta
        
        # make updates
        x = x + alpha*p
        r = r - alpha*q
        
        # update s
        s = constructSolutions(C_y=r, D=D, E=E, F=F, A=A_sample)
        
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