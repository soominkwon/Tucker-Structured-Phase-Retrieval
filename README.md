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


## Solution Example
<p align="center">
  <a href="url"><img src="https://github.com/soominkwon/Tucker-Structured-Phase-Retrieval/Video_Results/plane_recovery.gif" align="left" height="300" width="300" ></a>
</p>

## Comments
The algorithms AltMinLowRaP and AltMinTrunc (LRPR2) implementations are available in a different repository:

AltMinLowRaP: https://github.com/soominkwon/Provable-Low-Rank-Phase-Retrieval 

AltMinTrunc: https://github.com/soominkwon/Low-Rank-Phase-Retrieval
