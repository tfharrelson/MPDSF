***********************************
Features: Plotting Reach Curves 
***********************************

MPDSF is also equipped to calculate the reach curves for the implemented dark matter models. The reach is defined as the minimum cross-section required to observe 3 scattering events in a year when using 1 kg of the crystal to scatter dark matter. The metric makes no assumptions about our ability to actually detect it, it just counts the number of times dark matter is predicted to scatter off of the material. The details about the reach calculation are in these papers: `here <https://arxiv.org/abs/1910.08092>`_ and `here <https://arxiv.org/abs/1910.10716>`_

The reach calculator is set by putting the tag ``reach: True`` into the input file. This calculates the reach for the dark matter models set in the input file. The calculation is accelerated with Numba and MPI to make it relatively fast on a supercomputing cluster; I would not recommend running this on a personal computer for anything other than test calculations. Full convergence requires a supercomputer. The results are stored in the HDF5 dictionary under the key called ``reach``. I know, how very clever of me.
