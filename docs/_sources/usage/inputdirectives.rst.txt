.. _input_commands:

*****
Input Directives
*****

Below is a list of all input commands/directives along with descriptions:
        * **poscar** - name of the VASP POSCAR file.
        * **fc2** - Name of the 2nd order force constants file made from Phonopy. It is usually called ``FORCE_SETS``.
        * **fc3** - Name of the 3rd order force sets file made by Phono3py. It is usually called ``FORCES_FC3``.
        * **disp** - Name of the displacements file created by Phono3py/Phonopy in YAML format. It is sometimes called ``disp.yaml``. If not this, it is likely named something very similar.
        * **mesh** - A comma separated list representing the discretization of the Monkhorst-Pack grid used in the phonon calculations. The format of the tag must be like ``mesh: [ 5, 5, 5 ]``, in which the numbers can change, but the placement of the brackets and commas cannot.
        * **supercell** - A comma separated list representing the size of the supercell created to calculate the inter-atomic forces used in the phonon calculations. Must be of the form ``supercell: [ 2, 2, 2 ]``, and the numbers of the supercell must match what is in the ``disp.yaml`` file, and must accurately represent the size of the DFT calculations performed.
        * **max_energy** - The maximum energy (in THz) to be considered in the multiphonon calculation.
        * **delta_e** - The spacing between the discrete energies/frequencies.
        * **overtones** - The number of contact-type multiphonon interactions to consider in the calculation. For example, if 5 is input here, then all up to 5-phonon contact interactions will be considered. That is, the code will consider scattering events that generate 5 distinct phonons as valid.
        * **output** - The name of the output HDF5 file. It is technically allowed to specify a ``.txt`` file, but this is currently deprecated, and there are not any immediate plans to make this feature usable in the future. As such, specify ``hdf5`` files only.
        * **Gmesh** - The number of reciprocal lattice vectors to consider in addition to the q-points specified by the ``mesh`` tag. This tag is part of building the q-points for which the dynamic structure factor is calculated. The format must be similar to ``Gmesh: [ 2, 2, 2 ]``. In this example, the maximum reciprocal lattice vector is 1 (in reduced units) in each direction of the Brillouin zone. This is because the ``2`` value in the example is a non-inclusive maximum to the reciprocal lattice vector. Therefore, the final q-points array will include points like ``[ -1, 0, 0 ]``, ``[ 0, 0, 0 ]``, and ``[ 1, 0, 0 ]``. 
        * **shift** - You can change the center of you Monkhorst-Pack grid with this command. It must be of the form ``[ 0.1, 0.1, 0.1 ]``. Of course, the numbers in this example can change however you see fit.
        * **qmax** - Specify the maximum q-point magnitude (in reduced coordinates) for which dynamic structure factor data is written. This command is useful if you need to calculate the dynamic structure factor for a very large/dense grid, but do not need the high q-point values.
        * **mpi** - Boolean flag specifying if the code is being run with an MPI directive like ``srun`` or ``mpiexec``.
        * **born** - Name of the BORN file containing the Born effective charges and dielectric tensor. This file is generally good to calculate for nearly all materials, except for elemental solids. This file is also necessary for dark photon calculations.
        * **dark_photon** - Boolean flag specifying if you want the differential scattering rate for a dark photon mediated process to be calculated instead of the dynamic structure factor. The internal components are re-weighted according to expressions related to the Frohlich Hamiltonian before being summed.
        * **scalar_mediator** - Boolean flag specifying if you want the differential scattering rate for a light/heavy scalar mediated process to be calculated instead of the dynamic structure factor. The internal components are re-weighted by the atomic mass before being summed. Specification of light vs heavy mediators is in the ``med`` tag.
        * **med** - Tag specifying whether the scalar mediator is ``"light"`` or ``"heavy"``.
        * **param_lorentzian** - Boolean tag specifying if the anharmonic processes are parametrized by Lorentzian functions with widths given by the inverse phonon lifetimes.
        * **nofold_BZ** - Boolean tag specifying that the full Brillouin zone sampling of q-points is to be calculated, ignoring all symmetries. The reason for setting up the code this way is because it is recommended to fold Brillouin zones using the symmetries of the crystal structure to save memory and time. Only users who really want an unfolded Brillouin zone should use this tag.
        * **lowq_scaling** - Inclusion of low-q-point scaling characteristics of dynamic structure factor. The way this is done is through linear interpolation of phonon eigenvalues, eigenvectors, and imaginary self-energies, and the sampling points are chosen by dividing each q-point in the 1st Brillouin zone by the mesh (element-wise) to sample small q-points very close to the Gamma point. The use of this tag is strongly recommended for anyone doing dark matter calculations.
        * **no_anh** - Boolean flag directing the code to ignore the anharmonic processes even if the ``fc3`` tag is used.
        * **reach** - Boolean flag directing the code to calculate the dark matter reach for the material in addition to the differential scattering rates. The reach is defined as the minimum cross-section (in cm^2) to observe 3 dark matter scattering events in a year for 1 kg of the material.
