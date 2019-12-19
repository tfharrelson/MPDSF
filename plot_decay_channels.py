import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
import yaml_phonons as ph
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
    return grid_points, q_points;

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

def create_decay_spectrum(pp, triplets, triplet_map, grid_points, frequencies, delta_e, temperature, thresh, phonons=[], q_points=[]):
    max_e = max(frequencies) * 3.0
    num_bins = int(np.ceil(max_e / delta_e))
    spectrum = np.zeros(num_bins)
    pp_shape = pp.shape
    num_branches = pp_shape[-1]
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
                #print('qpoint1 =', phonons.qpoints[q1_index])
                #print('qpoint2 =', phonons.qpoints[q2_index])
                #print('qpoint3 =', phonons.qpoints[q3_index])

                #print('qpoint1 from ir file =', q_points[q1_index])
                #print('qpoint2 from ir file =', q_points[q2_index])
                #print('qpoint3 from ir file =', q_points[q3_index])

                combined_index_1 = get_combined_index(q1_index, pp_shape[1] - 1, num_branches)
                combined_index_2 = get_combined_index(q2_index, b2, num_branches)
                combined_index_3 = get_combined_index(q3_index, b3, num_branches)

                for i in range(len(pp_elements)):
                    if check_frequencies(frequencies[combined_index_1], frequencies[combined_index_2],
                                         frequencies[combined_index_3], thresh):
                        spectrum += pp_elements[i] / 2 * create_delta(frequencies[combined_index_2], delta_e, max_e) * \
                                    (compute_occupation_number(frequencies[combined_index_2], temperature) +
                                     compute_occupation_number(frequencies[combined_index_3], temperature) + 1)
                        spectrum += pp_elements[i] / 2 * create_delta(frequencies[combined_index_3], delta_e, max_e) * \
                                    (compute_occupation_number(frequencies[combined_index_2], temperature) +
                                     compute_occupation_number(frequencies[combined_index_3], temperature) + 1)
                        print('delta 1 satisfied!')
                    elif check_frequencies(frequencies[combined_index_1], -frequencies[combined_index_2],
                                           frequencies[combined_index_3], thresh):
                        spectrum += pp_elements[i] / 2 * create_delta(frequencies[combined_index_2], delta_e, max_e) * \
                                    (compute_occupation_number(frequencies[combined_index_2], temperature) -
                                     compute_occupation_number(frequencies[combined_index_3]), temperature)
                        spectrum += pp_elements[i] / 2 * create_delta(frequencies[combined_index_3], delta_e, max_e) * \
                                    (compute_occupation_number(frequencies[combined_index_2], temperature) -
                                     compute_occupation_number(frequencies[combined_index_3]), temperature)
                        print('delta 2 satisfied!')
                    elif check_frequencies(frequencies[combined_index_1], frequencies[combined_index_2],
                                           -frequencies[combined_index_3], thresh):
                        spectrum += pp_elements[i] / 2 * create_delta(frequencies[combined_index_2], delta_e, max_e) * \
                                    (-compute_occupation_number(frequencies[combined_index_2], temperature) +
                                     compute_occupation_number(frequencies[combined_index_3]), temperature)
                        spectrum += pp_elements[i] / 2 * create_delta(frequencies[combined_index_3], delta_e, max_e) * \
                                    (-compute_occupation_number(frequencies[combined_index_2], temperature) +
                                     compute_occupation_number(frequencies[combined_index_3]), temperature)
                        print('delta 3 satisfied!')
                    else:
                        print('No delta functions satisfied...')
                        print('frequency1 =', frequencies[combined_index_1])
                        print('frequency2 =', frequencies[combined_index_2])
                        print('frequency3 =', frequencies[combined_index_3])
                """
                for i in range(len(pp_elements)):
                    spectrum += pp_elements[i] * create_delta(frequencies[combined_index_2] +
                                                              frequencies[combined_index_3], delta_e, max_e) * \
                                (compute_occupation_number(frequencies[combined_index_2], temperature) +
                                 compute_occupation_number(frequencies[combined_index_3], temperature) + 1)
                    spectrum += pp_elements[i] * create_delta(frequencies[combined_index_3] -
                                                              frequencies[combined_index_2], delta_e, max_e) * \
                                (compute_occupation_number(frequencies[combined_index_2], temperature) -
                                 compute_occupation_number(frequencies[combined_index_3], temperature))
                    spectrum += pp_elements[i] * create_delta(frequencies[combined_index_2] -
                                                              frequencies[combined_index_3], delta_e, max_e) * \
                                (-compute_occupation_number(frequencies[combined_index_2], temperature) +
                                 compute_occupation_number(frequencies[combined_index_3], temperature))
                """
    print('frequency 1 =', frequencies[combined_index_1])
    return spectrum

def smear_spectrum(spectrum, sigma, delta_e):
    n_sigma = 10
    num_bins_half = int(np.ceil(n_sigma * sigma / delta_e))
    e_vals = np.arange(-num_bins_half, num_bins_half + 1) * delta_e
    gaussian = [1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-e**2 / (2 * sigma**2)) for e in e_vals]
    return signal.fftconvolve(spectrum, gaussian, mode='full')[num_bins_half:(num_bins_half + len(spectrum))] * delta_e

yaml_file = sys.argv[1]
pp_file = sys.argv[2]
ir_grid_file = sys.argv[3]
temperature = 4.0

pp_hdf5 = h5py.File(pp_file)
pp = np.array(pp_hdf5['pp'])
triplets = np.array(pp_hdf5['triplet'])
triplet_map = np.array(pp_hdf5['triplet_map'])

phonons = ph.Dyn_System(yaml_file)

delta_e = 0.01
grid_points, q_points = import_ir_grid(ir_grid_file)

thresh = compute_e_diff(phonons)
spectrum = create_decay_spectrum(pp, triplets, triplet_map, grid_points, phonons.frequencies, delta_e, temperature,
                                 thresh, phonons, q_points)

conv_THz_to_meV = 4.13567

spectrum = smear_spectrum(spectrum, 0.5 / conv_THz_to_meV, delta_e)
print('phonon energy =', 7.6463566 * conv_THz_to_meV)
plt.plot(np.linspace(0, len(spectrum) * delta_e, len(spectrum)) * conv_THz_to_meV, spectrum)
plt.xlabel('Energy (meV)')
plt.xlim([0, 50])
plt.show()
