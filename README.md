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

![plane_recovery](https://user-images.githubusercontent.com/43144680/135666576-7c6764b7-7d87-40f3-80b4-bdc08413e272.gif)

![mouse_recovery](https://user-images.githubusercontent.com/43144680/135897599-ee760697-bf87-4c23-9c7c-b9af312d6eb6.gif)


## Comments
The algorithms AltMinLowRaP and AltMinTrunc (LRPR2) implementations are available in a different repository:

AltMinLowRaP: https://github.com/soominkwon/Provable-Low-Rank-Phase-Retrieval 

AltMinTrunc: https://github.com/soominkwon/Low-Rank-Phase-Retrieval

A live reconstruction of the videos from TSPR is available: https://github.com/soominkwon/Tucker-Structured-Phase-Retrieval/tree/main/Video_Results
