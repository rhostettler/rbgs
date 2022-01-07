# Rao-Blackwellized Gaussian Filtering and Smoothing
This is an example Matlab implementation of the Rao-Blackwellized Gaussian 
filtering and smoothing algorithms introduced in:

* R. Hostettler and S. Särkkä, “Rao–Blackwellized Gaussian smoothing,” IEEE
  Transactions on Automatic Control, vol. 64, no. 1, pp. 302–309, January 2019.
  
  [[Link](https://doi.org/10.1109/TAC.2018.2828087)]
  [[PDF](http://hostettler.co/assets/publications/2019-tac.pdf)]

The numerical example is implemented in the file `rbgs_example.m`, while the
folder `lib` contains the algorithm implementation, helper functions, and
implementations of the compared algorithms.

Known issues:
* Iterative variants are not implemented.
* Variants for non-additive noise are not implemented.
* Defaults for the sigma-points in `gf`, `gf_*`, `rbgf`, and `rbgf_*` are
  missing.

