# ShallowCircuitClassicalShadows

This repository contains several codes to do efficient shallow circuit tomography on a classical computer as explained in https://arxiv.org/abs/2209.02093. 

* "CST_MPS_Paper_Plotmaker.ipynb" is an iPython notebook that has some relevant examples for how to make the figures in https://arxiv.org/abs/2209.02093. In particular, it has some codes for determining the shadow norms from the entanglement feature and reconstruction data, and how to do prediction on stabilizer states using an early version of the PyClifford package.

* "reconstruction-mps-short.py" and "HaarReconstructionChannel.py" demonstrate how to generate the reconstruction coefficients for a finite, brickwall circuit.  "HaarReconstructionMPS*.pt" contains the local MPS tensors.

* "MPS.py" and "EF_MPS_utils.py" contain methods that are used to do tensor network simulations.

* "base" contains several packages for doing stabilizer state simulation. We suggest using the PyClifford package found on github here: https://hongyehu.github.io/PyCliffordPages/intro.html

* "data" contains the MPS tensors for the reconstruction coefficients used in the paper.

* "fidelity-job-2.py" contains some code designed to be run on a cluster for doing fidelity estimations. 

* "stabilizer_mps_utils.py" contains several codes for converting between the stabilizer tableau representation and a tensor network representation for stabilizer states.

* "ef-mps-solver.py" computes an MPS representation for the entanglement feature of a finite depth circuit. It works by solving for the MPS using TEBD at a large bond dimension D1, and then we simplify the resulting state to an MPS of small bond dimension D2, using "vMPSSolver.py".

* "EFMPS*.pt" contains the local tensor for the entanglement feature of various short depth brick wall circuits.

Please contact a1akhtar AT ucsd.edu for questions about the codes.

