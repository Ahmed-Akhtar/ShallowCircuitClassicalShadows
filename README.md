# ShallowCircuitClassicalShadows

This repository contains several codes to do efficient shallow circuit tomography on a classical computer as explained in https://arxiv.org/abs/2209.02093. 

* "CST_MPS_Paper_Plotmaker.ipynb" is an iPython notebook that has some relevant examples for how to make the figures in https://arxiv.org/abs/2209.02093. 

* "reconstruction-mps-short.py" and "HaarReconstructionChannel.py" demonstrate how to generate the reconstruction coefficients for a finite, brickwall circuit.

* "MPS.py" and "EF_MPS_utils.py" contain methods that are used to do tensor network simulations.

* "base" contains several packages for doing stabilizer state simulation. We suggest using the PyClifford package found on github here: https://hongyehu.github.io/PyCliffordPages/intro.html

* "data" contains the MPS tensors for the reconstruction coefficients used in the paper.
