# MPDSF

The purpose of this code is to calculate the phonon dynamic structure factor from first principles DFT calculations. The only input files supported right now are from Phonopy and Phono3py. For full details about the code, see the documentation [here](https://tfharrelson.github.io/MPDSF/). The standard usage of the code starts from the third-order force constants calculated from Phono3py, a VASP POSCAR, a Phono3py disp file (`*_disp.yaml`), and (optionally) a BORN file generated from Phonopy. From these files, the code calculates the multi-phonon dynamic structure factor that includes 3-phonon anharmonic processes and n-phonon contact interactions. General n-phonon contact interactions are made computationally tractable by expressing the multiphonon interactions as a convolution operation, and using the convolution theorem to accelerate the calculation by using FFT's. Essentially, we FFT the reciprocal-space objects to real space, take the direct product, and inverse FFT back to reciprocal space to get the n-phonon interaction contribution. In this case, the inputs are the `n-1` phonon interaction and the 1-phonon interaction functions, which, when convolved, yield the n-phonon interaction. Through the use of FFT's, the convolution operation is not the bottleneck in the code. 

The output is written as an HDF5 file which stores the dynamic structure factor calculated at each q-point requested. The end goal of using this code is to calculate a more precise dynamic structure factor for neutron scattering, or to calculate the scattering response for a variety of nucleon-based dark matter models. For the latter case, the currently implemented dark matter scattering models are the light and heavy scalar mediator, and the dark-photon mediated interactions. For details on the functional forms of the interactions, see this [paper](https://arxiv.org/abs/1910.10716).

The code also contains general object-oriented modules that handle different phonon related parameters, such as eigenvalues, eigenvectors, imaginary self energies, gamma functions (precursor function from which imaginary self energies are computed). These modules have been used to create an athermal phonon transport model that calculates the rate of phonon transfer across an interface from a single athermal phonon generated from a dark photon scattering event. The code for this calculation is contained in the `scripts` folder. The general phonon modules are in `src`.

## Installation

Currently installation via pip is not supported, but `requirements.yml` provide all the necessary dependences to create a working conda environment. To do so, we recommend running `conda create -f requirements.yml`, which will create a conda environment called `multiphonon` which contains all packages needed to run MPDSF. Also, there will be a working version of the code on Docker soon.

## Usage

After installation, the code is best run using an input file in YAML format. Example input files are in the `examples` folder. Once a suitable input file is created, running the code is executed via `\path\to\executable\multiphonon_dsf.py -i input_file.yaml` where the path is either explicitly typed, or the path is added to the PATH variable before running the `multiphonon_dsf.py` command.

To specify different dark matter models, change the `med`, `dark_photon`, and `scalar_mediator` tags to the appropriate values. The `dark_photon` and `scalar_mediator` tags are Boolean values, and the `med` tag is specified with either `heavy` or `light`. To calculate the reach automatically (defined as the minimum dark matter cross-section that would experience 3 scattering events in a kg target over a year of observation), set `reach: True`.

### Tips

If you are interested in calculating the dark matter scattering cross-section, then large q-grids must be used to converge your results, and calculating the dynamic structure factor on a supercomputing cluster is advised. The code is parallelized for HPC calculations if the tag `mpi: True` is supplied in the input file. For this case, the type of dark matter interaction changes the type of q-grid one should request. For both massless-mediated models (light scalar and dark photon mediated), keeping the q-grid in the 1st Brillouin zone is advisable, as well as turning on the `low_q_scaling: True` feature because the most important q-points to sample are near the gamma point of the Brilloin zone. For the heavy scalar mediator, one should request q-points far outside the 1st Brillouin zone. This is set by the `Gmesh` option which specifies the mesh of reciprocal lattice vectors (which we call G-vectors) to go along with the `mesh` tag which specifies the way in which the first Brillouin zone is discretized (which we call k-points). Therefore, each q-point is specified by a G-point and a k-point; the k-point is a cell-periodic number and is not defined outside the first Brillouin zone. Thus, to get proper convergence of heavy scalar mediator interactions, one should use a reduced `mesh` tag (e.g. `[ 7, 7, 7 ]`), and a large `Gmesh` tag (e.g. `[ 21, 21, 21 ]`). The usage of `low_q_scaling: True` is also recommended for this interaction. 

## Citing us

The multiphonon code paper is still under revision, check back for updates soon.

Also, our paper on the athermal phonon transport model is still under revision as well. 
