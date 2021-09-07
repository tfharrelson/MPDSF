.. _quick_start:

*****
Quick Start Guide
*****

Welcome to MPDSF! This is the code for calculating the multi-phonon dynamic structure factor completely from first-principles calculations. The code is built using the Phonopy/Phono3py API, and currently assumes you are using VASP as your DFT solver. If you have a need for calculating the phonon dynamic structure factor complete with so-called contact and anharmonic interactions, then this is the code for you. Applications for this code are the calculation of dark matter scattering rates, and neutron scattering responses. The code utilizes the magic of fast Fourier transforms to speed up the contact interaction calculation by casting the calculation as a convolution operation.

While the code can be run many different ways, by far the easiest is by creating an input file in YAML format. Examples are in the 'examples' folder (I know shocking). I would strongly recommend taking one of these files and modifying it for your needs. 

The full list of input commands are described at the ``Input Directives`` section.

For dark matter enthusiasts, the input commands of most interest will be ``med``, ``scalar_mediator_flag``, ``dark_photon_flag``, and ``reach``. Be sure to read up on those!

For the neutron scattering enthusiasts, the input command of the most interest will be ``neutron_flag``.

Once an input file is created the code can be run by calling::
        
        python3 mpdsf.py -i my_input_file.yaml

Outputs will be written to an HDF5 file in the same directory that the above command was run. This output file can get quite large for very well-converged calculations. Since python has a nice HDF5 library called ``h5py``, which effectively turns each HDF5 file into a python dictionary, you can check the contents of your output by ::

        import h5py
        f = h5py.File('output_file.hdf5', 'r')
        print(f.keys())
