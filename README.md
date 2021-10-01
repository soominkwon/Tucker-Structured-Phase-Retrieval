# Tucker-Structured-Phase-Retrieval

Tucker-Structured Phase Retrieval (TSPR) is an algorithm that can recover a tensor from phase-less linear measurements of its frontal slices. This algorithm factorizes the tensor using the Tucker decomposition and solves for the Tucker factors using conjugate gradient least squares.

The implementation details are available here: https://soominkwon.github.io/Publications/Papers/ICASSP_Supplementary_Materials.pdf

For more information about TSPR:

## Programs
The following is a list of which algorithms correspond to which Python script:

* tucker_cgls_D.py - Customized conjugate gradient least squares (CGLS) solver for updating factor D
* tucker_cgls_E.py - Customized conjugate gradient least squares (CGLS) solver for updating factor E
* tucker_cgls_core.py - Customized conjugate gradient least squares (CGLS) solver for updating core factor
* tucker_phase_retrieval.py - Implementation of Tucker-Structured Phase Retrieval (TSPR)
* reshaped_wirtinger_flow.py - Implementation of RWF
* higher_order_SVD.py - Implementation of higher-order SVD
* reconstruct_tensor.py - Code for reconstructing a third-order given Tucker factors
* sample_run.py - Example run file

## Tutorial
This tutorial can be found in sample_run.py:

```
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
```

## Solution Example
<p align="center">
  <a href="url"><img src="https://github.com/soominkwon/Tucker-Structured-Phase-Retrieval/Video_Results/plane_recovery.mp4" align="left" height="300" width="300" ></a>
</p>

## Comments
The algorithms AltMinLowRaP and AltMinTrunc (LRPR2) implementations are available in a different repository:

AltMinLowRaP: https://github.com/soominkwon/Provable-Low-Rank-Phase-Retrieval 

AltMinTrunc: https://github.com/soominkwon/Low-Rank-Phase-Retrieval
