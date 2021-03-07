import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
from src import yaml_phonons as ph
from scipy import signal
import scipy.constants as const
import re

def get_combined_index(q_index, branch_index, num_branches):
    # purpose is to get single index from both q_index and branch_index
    # single index is used in phonopy outputs, but indices are separated in phono3py outputs
    combined_index = int(q_index * num_branches + branch_index)
    return combined_index

def import_ir_grid(ir_grid_file):
    file_reader = open(ir_grid_file, 'r')
    grid_points = np.array([])
    q_points = np.empty([0, 3])
    p = re.compile('[+-]?(\d+\.\d+)')
    for line in file_reader:
        if line[:12] == '- grid_point':
            gp = float(line.split()[2])
            grid_points = np.append(grid_points, gp)
        if line[2:9] == "q-point":
            qpt = np.array([p.findall(line)]).astype(np.float)
            q_points = np.append(q_points, qpt, 0)
    return grid_points, q_points

def create_delta(energy, delta_e, max_e):
    # units in meV
    num_bins = int(np.ceil(max_e/delta_e))
    delta_fcn = np.zeros(num_bins)

    if energy < 0:
        return delta_fcn

    e_bin_minus = int(np.floor(energy/delta_e))
    e_bin_plus = int(np.ceil(energy/delta_e))

    alpha_minus = np.abs(e_bin_minus * delta_e - energy) / delta_e
    alpha_plus = np.abs(e_bin_plus * delta_e - energy) / delta_e

    delta_fcn[e_bin_minus] = (1 - alpha_minus) / delta_e
    delta_fcn[e_bin_plus] = (1 - alpha_plus) / delta_e
    return delta_fcn

def check_frequencies(freq_1, freq_2, freq_3, threshold):
    #threshold = 1e-6
    if (freq_1 - threshold/2) < (freq_2 + freq_3) and (freq_1 + threshold/2) > (freq_2 + freq_3):
        return True
    else:
        return False

def compute_occupation_number(frequency, temperature):
    return (np.exp(const.hbar * frequency * 10**12 / (const.Boltzmann * temperature)) - 1)**-1

def compute_e_diff(phonons):
    q_index = phonons.natoms * 3
    freq_diffs = phonons.frequencies[q_index:q_index+3]
    return np.amax(freq_diffs)

def get_gridpoints(triplet_map):
    return np.unique(triplet_map)

def gaussian(x, sigma):
    return 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-1.0 * x ** 2 / (2 * sigma ** 2))

def create_gaussian(mu, sigma, delta_e, max_e):
    num_bins = int(np.ceil(max_e / delta_e))
    frequencies = np.arange(num_bins)*delta_e
    return gaussian(frequencies - mu, sigma)

def create_decay_spectrum(pp,
                          triplets,
                          triplet_map,
                          frequencies,
                          delta_e,
                          temperature,
                          sigma=0.1,
                          grid_points=None,
                          gammas=None,
                          mode_vels=None,
                          min_freq=0.01):
    """

    :param pp: A set of phonon-phonon interaction parameters (equal to |matrix element|^2) for a specific grid-point
    that maps to a specific q-point. The shape is [num_triplets, num_branches, num_branches, num_branches]
    :param triplets: List of triplet sets specified by grid-point indices. E.g. triplets[0, :] = [0, 1, 5] means that
    grid-points 0, 1, and 5 form an allowed triplet of q-points
    :param triplet_map: List that maps all triplets to irreducible points in Brillouin zone. In the triplet list, the
    first index is specified by the grid-point, then the second index is free, and the third is fixed by the crystal
    momentum conservation equation q_3 = G - q_1 - q_2, where G is a recip. lattice vector.
    :param grid_points:
    :param frequencies:
    :param delta_e:
    :param temperature:
    :param sigma: Gaussian broadening factor that smooths convergence for integration over BZ
    :param gammas: Data structure containing the relaxation rates for each phonon in the irr. Brillouin zone. Shape is
    [num_temperatures, q-points, branches]. To index correct q-points, need to map grid-points to q-points
    :param min_freq: Ignore all phonons below this frequency threshold:
    :return:
    """
    max_e = max(frequencies) * 3.0
    num_bins = int(np.ceil(max_e / delta_e))
    spectrum = np.zeros(num_bins)
    property_spectrum = np.zeros(num_bins)
    pp_shape = pp.shape
    num_branches = pp_shape[-1]
    if grid_points is None:
        grid_points = get_gridpoints(triplet_map)
    # set max lifetime to be comparable to largest lifetime allowed by isotopic variation
    max_lifetime = 1e6
    min_factor = 1e-4 / num_bins
    for tr in range(pp_shape[0]):
        for b2 in range(pp_shape[2]):
            for b3 in range(pp_shape[3]):
                # tr = triplet index; b2 = branch index 2; b3 = branch index 3
                pp_elements = pp[tr, (pp_shape[1]-3):, b2, b3]   # get all three pp elements for optic branches
                triplet = triplets[tr, :]

                gp1_index = triplet_map[triplet[0]]
                gp2_index = triplet_map[triplet[1]]
                gp3_index = triplet_map[triplet[2]]

                q1_index = np.where(grid_points == gp1_index)[0][0]
                q2_index = np.where(grid_points == gp2_index)[0][0]
                q3_index = np.where(grid_points == gp3_index)[0][0]

                combined_index_1 = get_combined_index(q1_index, pp_shape[1] - 1, num_branches)
                combined_index_2 = get_combined_index(q2_index, b2, num_branches)
                combined_index_3 = get_combined_index(q3_index, b3, num_branches)
                #print('pp_shape =', pp.shape)
                if frequencies[combined_index_2] < min_freq:
                    continue
                if frequencies[combined_index_3] < min_freq:
                    continue

                if gammas is not None:
                    if gammas[0, q2_index, b2] == 0:
                        property_2 = max_lifetime
                    else:
                        property_2 = 1 / (4 * np.pi * gammas[0, q2_index, b2])

                    if gammas[0, q3_index, b3] == 0:
                        property_3 = max_lifetime
                    else:
                        property_3 = 1 / (4 * np.pi * gammas[0, q3_index, b3])

                    if mode_vels is not None:
                        property_2 *= np.linalg.norm(mode_vels[q2_index, b2, :])
                        property_3 *= np.linalg.norm(mode_vels[q3_index, b3, :])
                elif mode_vels is not None:
                    property_2 = np.linalg.norm(mode_vels[q2_index, b2, :])
                    property_3 = np.linalg.norm(mode_vels[q3_index, b3, :])
                for i in range(len(pp_elements)):
                    combined_index_1 = get_combined_index(q1_index, pp_shape[1] - 1 - i, num_branches)
                    #if check_frequencies(frequencies[combined_index_1], frequencies[combined_index_2],
                    #                     frequencies[combined_index_3], thresh):
                    int_factor = gaussian(frequencies[combined_index_1] - frequencies[combined_index_2]
                                          - frequencies[combined_index_3], sigma=sigma)
                    #print('int_factor=', int_factor)
                    if int_factor > min_factor:
                        spectrum += pp_elements[i] / 2 * create_gaussian(frequencies[combined_index_2], sigma, delta_e, max_e) * \
                                    (compute_occupation_number(frequencies[combined_index_2], temperature) +
                                     compute_occupation_number(frequencies[combined_index_3], temperature) + 1) * int_factor
                        spectrum += pp_elements[i] / 2 * create_gaussian(frequencies[combined_index_3], sigma, delta_e, max_e) * \
                                    (compute_occupation_number(frequencies[combined_index_2], temperature) +
                                     compute_occupation_number(frequencies[combined_index_3], temperature) + 1) * int_factor
                        if gammas is not None:
                            property_spectrum += pp_elements[i] / 2 * create_gaussian(frequencies[combined_index_2], sigma, delta_e,
                                                                          max_e) * \
                                        (compute_occupation_number(frequencies[combined_index_2], temperature) +
                                         compute_occupation_number(frequencies[combined_index_3], temperature) + 1) * \
                                             property_2 * int_factor
                            property_spectrum += pp_elements[i] / 2 * create_gaussian(frequencies[combined_index_3], sigma, delta_e,
                                                                          max_e) * \
                                        (compute_occupation_number(frequencies[combined_index_2], temperature) +
                                         compute_occupation_number(frequencies[combined_index_3], temperature) + 1) * \
                                             property_3 * int_factor
                    #elif check_frequencies(frequencies[combined_index_1], -frequencies[combined_index_2],
                    #                       frequencies[combined_index_3], thresh):
                    int_factor = gaussian(frequencies[combined_index_1] + frequencies[combined_index_2]
                                          - frequencies[combined_index_3], sigma=sigma)
                    if int_factor > min_factor:
                        spectrum += pp_elements[i] / 2 * create_gaussian(frequencies[combined_index_2], sigma, delta_e, max_e) * \
                                    (compute_occupation_number(frequencies[combined_index_2], temperature) -
                                     compute_occupation_number(frequencies[combined_index_3], temperature)) * int_factor
                        spectrum += pp_elements[i] / 2 * create_gaussian(frequencies[combined_index_3], sigma, delta_e, max_e) * \
                                    (compute_occupation_number(frequencies[combined_index_2], temperature) -
                                     compute_occupation_number(frequencies[combined_index_3], temperature)) * int_factor
                        if gammas is not None:
                            property_spectrum += pp_elements[i] / 2 * create_gaussian(frequencies[combined_index_2], sigma, delta_e,
                                                                          max_e) * \
                                        (compute_occupation_number(frequencies[combined_index_2], temperature) -
                                         compute_occupation_number(frequencies[combined_index_3], temperature)) * \
                                             property_2 * int_factor
                            property_spectrum += pp_elements[i] / 2 * create_gaussian(frequencies[combined_index_3], sigma, delta_e,
                                                                          max_e) * \
                                        (compute_occupation_number(frequencies[combined_index_2], temperature) -
                                         compute_occupation_number(frequencies[combined_index_3], temperature)) * \
                                             property_3 * int_factor
                    #elif check_frequencies(frequencies[combined_index_1], frequencies[combined_index_2],
                    #                       -frequencies[combined_index_3], thresh):
                    int_factor = gaussian(frequencies[combined_index_1] - frequencies[combined_index_2]
                                          + frequencies[combined_index_3], sigma=sigma)
                    if int_factor > min_factor:
                        spectrum += pp_elements[i] / 2 * create_gaussian(frequencies[combined_index_2], sigma, delta_e, max_e) * \
                                    (-compute_occupation_number(frequencies[combined_index_2], temperature) +
                                     compute_occupation_number(frequencies[combined_index_3], temperature)) * int_factor
                        spectrum += pp_elements[i] / 2 * create_gaussian(frequencies[combined_index_3], sigma, delta_e, max_e) * \
                                    (-compute_occupation_number(frequencies[combined_index_2], temperature) +
                                     compute_occupation_number(frequencies[combined_index_3], temperature)) * int_factor
                        if gammas is not None:
                            property_spectrum += pp_elements[i] / 2 * create_gaussian(frequencies[combined_index_2], sigma, delta_e,
                                                                          max_e) * \
                                        (-compute_occupation_number(frequencies[combined_index_2], temperature) +
                                         compute_occupation_number(frequencies[combined_index_3], temperature)) * \
                                             property_2 * int_factor
                            property_spectrum += pp_elements[i] / 2 * create_gaussian(frequencies[combined_index_3], sigma, delta_e,
                                                                          max_e) * \
                                        (-compute_occupation_number(frequencies[combined_index_2], temperature) +
                                         compute_occupation_number(frequencies[combined_index_3], temperature)) * \
                                             property_3 * int_factor
                    #else:
                        #print('No delta functions satisfied...')
                        #print('frequency1 =', frequencies[combined_index_1])
                        #print('frequency2 =', frequencies[combined_index_2])
                        #print('frequency3 =', frequencies[combined_index_3])
    print('frequency 1 =', frequencies[combined_index_1])
    if gammas is not None:
        return property_spectrum / (np.trapz(spectrum) * delta_e)
    else:
        return spectrum / (np.trapz(spectrum) * delta_e)

#def create_lifetime_spectrum

#def compute_dispersive_shifts():

def smear_spectrum(spectrum, sigma, delta_e):
    n_sigma = 10
    num_bins_half = int(np.ceil(n_sigma * sigma / delta_e))
    e_vals = np.arange(-num_bins_half, num_bins_half + 1) * delta_e
    gaussian = [1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-e**2 / (2 * sigma**2)) for e in e_vals]
    return signal.fftconvolve(spectrum, gaussian, mode='full')[num_bins_half:(num_bins_half + len(spectrum))] * delta_e

#yaml_file = sys.argv[1]
#pp_file = sys.argv[2]
#ir_grid_file = sys.argv[3]
#gamma_file = '/Volumes/GoogleDrive/My Drive/Python Scripts/si_232323_kappa_T4K.hdf5'
yaml_file = '/Volumes/GoogleDrive/My Drive/multiphonon/Si_files/si_232323_mesh.yaml'
pp_file = '/Volumes/GoogleDrive/My Drive/multiphonon/Si_files/si-pp-gamma.hdf5'
ir_grid_file = '/Volumes/GoogleDrive/My Drive/multiphonon/Si_files/si_232323_grid.yaml'
gamma_file = '/Volumes/GoogleDrive/My Drive/Python Scripts/si_232323_kappa_T4K.hdf5'

gaas_gamma = '/Volumes/GoogleDrive/My Drive/multiphonon/GaAs_files/gaas_kappa_232323_4K.hdf5'
gaas_grid = '/Volumes/GoogleDrive/My Drive/multiphonon/GaAs_files/gaas_232323_ir_grid.yaml'
gaas_mesh = '/Volumes/GoogleDrive/My Drive/multiphonon/GaAs_files/gaas_232323_mesh.yaml'
gaas_pp = '/Volumes/GoogleDrive/My Drive/multiphonon/GaAs_files/gaas_232323_pp_gamma.hdf5'

yaml_files = [gaas_mesh, yaml_file]
pp_files = [gaas_pp, pp_file]
ir_grid_files = [gaas_grid, ir_grid_file]
gamma_files = [gaas_gamma, gamma_file]
temperature = 4.0
labels = ['GaAs', 'Si']

#plt.style.use('tableau-colorblind10')
spectra = []

for yaml_file, pp_file, ir_grid_file, gamma_file, label in zip(yaml_files, pp_files, ir_grid_files, gamma_files, labels):

    pp_hdf5 = h5py.File(pp_file, 'r')
    pp = np.array(pp_hdf5['pp'])
    triplets = np.array(pp_hdf5['triplet'])
    triplet_map = np.array(pp_hdf5['triplet_map'])

    phonons = ph.Dyn_System(yaml_file)

    delta_e = 0.01
    grid_points, q_points = import_ir_grid(ir_grid_file)

    conv_THz_to_meV = 4.13567
    #thresh = compute_e_diff(phonons)
    sigma = 0.5 / conv_THz_to_meV
    # get imaginary self-energies from new file
    #gamma_file = '/Volumes/GoogleDrive/My Drive/multiphonon/GaAs_files/gaas_kappa_232323_4K.hdf5'
    gamma_h5 = h5py.File(gamma_file, 'r')
    gammas = np.array(gamma_h5['gamma'])
    #gammas=None
    mode_vels = np.array(gamma_h5['group_velocity'])
    #mode_vels = None
    spectrum = create_decay_spectrum(pp, triplets, triplet_map, phonons.frequencies, delta_e, temperature,
                                     sigma, gammas=gammas, mode_vels=mode_vels, grid_points=None)

    #spectrum = smear_spectrum(spectrum, 0.1/ conv_THz_to_meV, delta_e)

    print('phonon energy =', 7.6463566 * conv_THz_to_meV)
    plt.plot(np.linspace(0, len(spectrum) * delta_e, len(spectrum)) * conv_THz_to_meV, spectrum / conv_THz_to_meV,
             label=label)
    spectra.append(spectrum)
plt.xlabel('Energy (meV)')
plt.xlim([0, 60])
plt.yscale('log')
plt.ylim([1e-2, 1e10])
#plt.ylim([0, 0.16])
plt.legend()
plt.tight_layout()
plt.savefig('/Volumes/GoogleDrive/My Drive/Manuscripts/multiphonon/Figures/combined_decay_channels_mfp.eps', dpi=600)
plt.show()
