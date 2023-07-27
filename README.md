# PyRaTE - non-LTE spectral lines simulations

PyRaTE is a multilevel radiative transfer code designed to post-process astrochemical simulations. The code uses the escape probablity method to calculate the population densities of the species under consideration. The code can handle all projection angles and geometries and can also be used to produce mock observations of the Goldreich-Kylafis effect.

The code is written in Python using an embarrassingly parallel strategy and it relies on the YT analysis toolkit, mpi4py and numba.
