# Compute_Sqw

The purpose of this code is to calculate the phonon dynamic structure factor from first principles DFT calculations. The only input files supported right now are from Phonopy and Phono3py. The standard usage of the code starts from the third-order force constants calculated from Phono3py, a VASP POSCAR, a Phono3py disp file (`*_disp.yaml`), and (optionally) a BORN file generated from Phonopy. From these files, the code calculates the multi-phonon dynamic structure factor that includes 3-phonon anharmonic processes and n-phonon contact interactions. The output is written as an HDF5 file which stores the dynamic structure factor calculated at each q-point requested. The end goal of using this code is to calculate a more precise dynamic structure factor for neutron scattering, or to calculate the scattering response for a variety of nucleon-based dark matter models. For the latter case, the currently implemented dark matter scattering models are the light and heavy scalar mediator, and the dark-photon mediated interactions. For details on the functional forms of the interactions, see this [paper](https://arxiv.org/abs/1910.10716).

## Installation

Currently installation via pip is not supported, but `requirements.txt` provide all the necessary dependences to create a working conda environment. Also, there will be a working version of the code on Docker soon.

## Usage

After installation, the code is best run using an input file in YAML format. Example input files are in the `examples` folder. Once a suitable input file is created, running the code is executed via `\path\to\executable\multiphonon_dsf.py -i input_file.yaml` where the path is either explicitly typed, or the path is added to the PATH variable before running the `multiphonon_dsf.py` command.

To specify different dark matter models, change the `med`, `dark_photon`, and `scalar_mediator` tags to the appropriate values. The `dark_photon` and `scalar_mediator` tags are Boolean values, and the `med` tag is specified with either `heavy` or `light`. To calculate the reach automatically (defined as the minimum dark matter cross-section that would experience 3 scattering events in a kg target over a year of observation), set `reach: True`.

### Tips

If you are interested in calculating the dark matter scattering cross-section, then large q-grids must be used to converge your results, and calculating the dynamic structure factor on a supercomputing cluster is advised. The code is parallelized for HPC calculations if the tag `mpi: True` is supplied in the input file. For this case, the type of dark matter interaction changes the type of q-grid one should request. For both massless-mediated models (light scalar and dark photon mediated), keeping the q-grid in the 1st Brillouin zone is advisable, as well as turning on the `low_q_scaling: True` feature because the most important q-points to sample are near the gamma point of the Brilloin zone. For the heavy scalar mediator, one should request q-points far outside the 1st Brillouin zone. This is set by the `Gmesh` option which specifies the mesh of reciprocal lattice vectors (which we call G-vectors) to go along with the `mesh` tag which specifies the way in which the first Brillouin zone is discretized (which we call k-points). Therefore, each q-point is specified by a G-point and a k-point; the k-point is a cell-periodic number and is not defined outside the first Brillouin zone. Thus, to get proper convergence of heavy scalar mediator interactions, one should use a reduced `mesh` tag (e.g. `[ 7, 7, 7 ]`), and a large `Gmesh` tag (e.g. `[ 21, 21, 21 ]`). The usage of `low_q_scaling: True` is also recommended for this interaction. 

## Citing us
The paper is still under revision, check back for updates soon.
