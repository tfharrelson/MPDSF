import numpy as np
from scipy import signal
import sys
#import yaml_phonons as ph
import phonon_lifetimes
import math
import scipy.constants as const
import matplotlib.pyplot as plt
import spglib as spg
from numba import jit

"""
Goal of program is to compute S(q,w) with q-points (as vectors) as user-defined inputs.
Definition of S(q,w):
F_i(q,w) = \sum_j (q * u_{ij})^2 exp[-\sum_j (q * u_ij)^2] \delta(\hbar w - \hbar w_j)
S_i(q,w) = \sum_n conv^n/n! * F_i(q,w)
S(q,w) = \sum_i S_i(q,w)
which is perfectly defined for the "incoherent" approximation in neutron scattering.
Other words: this is the space and time Fourier transform of the self van-Hove correlation function
To include "coherent" effects (e.g. compute the full van-Hove correlation function):
f_i = \sum_j (q * u_{ij}) \delta(\hbar w - \hbar w_j)
DWF_i = exp[-\sum_j (q * u_ij)^2/2] 
F_ab(q,w) = conj(f_a) * f_b
S(q,w) = \sum_ab F_ab * DWF_a * DWF_b

What happens when you have two degenerate phonon modes (w_i == w_j)
    |(q * u_i) + (q * u_j)|^2 = |q*u_i|^2 + |q*u_j|^2 + conj(q*u_i) * (q*u_j) + c.c.
"""

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
    #if num_bins % 2 == 0:
    #    # even case
    #    delta_fcn[num_bins/2] = 1/(2 * delta_e)
    #    delta_fcn[num_bins/2 - 1] = 1/(2 * delta_e)
    #else:
    #    index = int(np.ceil(num_bins/2))
    #    delta_fcn[index] = 1 / delta_e
    return delta_fcn

def create_lorentzian(energy, gamma, delta_e, max_e):
    # create a Lorentzian function instead of a delta function for peaks broadened by anharmonicities
    # units of energy are THz for compatibility with phono3py

    num_bins = int(np.ceil(max_e / delta_e))
    lorentzian = np.zeros(num_bins)

    # check if energy is negative, return zero vector if true
    if energy < 0:
        return lorentzian

    # check if gamma is less than resolution of grid, if so return delta fcn instead of lorentzian
    if gamma < delta_e:
        return create_delta(energy, delta_e, max_e)

    # in an effort to speed up section
    n_sigma = 10
    #x_vals = np.arange(num_bins) * delta_e
    #lorentzian = np.array([(1 / np.pi * 0.5 * gamma / ((energy - x_val)**2 + (0.5 * gamma)**2))
    #                       if (x_val < energy + n_sigma * gamma) and (x_val > energy + n_sigma * gamma)
    #                       else 0.0 for x_val in x_vals])
    bin_spread = int(np.ceil(n_sigma * gamma / delta_e))
    e_bin = int(np.round(energy / delta_e))
    x_vals = np.arange(e_bin - bin_spread, e_bin + bin_spread + 1) * delta_e
    for i in range(len(x_vals)):
        lorentzian[i + e_bin - bin_spread] = 1 / np.pi * 0.5 * gamma / ((energy - x_vals[i])**2 + (0.5 * gamma)**2)
    return lorentzian

def compute_f_DWF(q_point, phonons, atomic_index):
    DWF = 0.0
    norm_constant = 1 / np.sum(phonons.weights)
    num_bands = phonons.eigvecs.shape[3]
    num_qpts = phonons.eigvecs.shape[2]
    for i in range(num_qpts):
        for j in range(num_bands):
            if phonons.frequencies[i] < 0:
                continue
            #q_index = int(np.floor(i / (3 * phonons.natoms)))
            #weight_index = i * num_bands + j
            DWF += np.abs(np.dot(q_point, phonons.eigvecs[atomic_index, :, i, j]))**2 * phonons.weights[i]
    return np.exp(-norm_constant * DWF/2)

def compute_f_i(q_point, phonons, atomic_index, coh_flag=False):
    #num_bins = int(np.floor(max_e/delta_e))
    #f_i = np.zeros(num_bins, dtype=np.complex)
    if coh_flag:
        norm_constant = 1.0
    else:
        #norm_constant = 1.0
        norm_constant = np.sqrt(1 / sum(phonons.weights))
    num_eigs = phonons.eigvecs.shape[2] * phonons.eigvecs.shape[3]
    f_i = np.zeros(num_eigs, dtype=np.complex)
    for i in range(len(phonons.frequencies)):
        #print('qpt vdot eigvec = '), print(np.sum(np.vdot(q_point, phonons.eigvecs[atomic_index, :, i])))
        #print('i =', i)
        q_index = np.floor(i / (3 * phonons.natoms)).astype(np.int)
        #print('qindex =', q_index)
        band_index = i % (3 * phonons.natoms)
        #f_i += phonons.weights[q_index] * np.vdot(q_point, phonons.eigvecs[atomic_index, :, i]) \
        #       * create_delta(phonons.frequencies[i], delta_e, max_e)
        f_i[i] = np.sqrt(phonons.weights[q_index]) * np.vdot(q_point, phonons.eigvecs[atomic_index, :, q_index, band_index])
    return f_i * norm_constant#/ np.sqrt(np.sum(phonons.weights) * 8 * const.pi**3)

def circ_conv(signal, kernel):
    # purpose is to perform a circular convolution (e.g. signal is periodic) of a 3-dimensional object
    # in this case, the 3-d object is a function proportional to phonon eigenvector as a function of q in the BZ
    #TODO Incorporate my own padding to the frequency dimension of the circular convolution; use appropriate tags
    return np.fft.ifftn(np.fft.fftn(signal) * np.fft.fftn(kernel))

def convolve_f_i(f_i, num_overtones, delta_e, coh_flag=False, dxdydz=[]):

    #norm_constant = 1 / sum(phonons.weights)
    norm_constant = 1
    curr_f = f_i #/ sum(phonons.weights)
    f_i = f_i * norm_constant
    total_f_i = curr_f
    # norm_constant = 1 / 8
    #norm_constant = 1 / (8 * sum(phonons.weights))
    for i in range(1, num_overtones):
        #print('n ='), print(i+1)
        if i > 1:
            curr_f = curr_f * norm_constant
        #print('int of curr_f'), print(np.trapz(curr_f) * delta_e)
        if coh_flag:
            if i == 1:
                #TODO implement change in this magic number if this normalization scheme works
                #curr_f *= 729
                curr_f *= 1
            curr_f = circ_conv(curr_f, f_i) * delta_e * dxdydz
        else:
            curr_f = signal.fftconvolve(curr_f, f_i, mode='full')[:len(f_i)] * delta_e
        #print('int of curr_f'), print(np.trapz(curr_f) * delta_e)
        #total_f_i += curr_f / (np.sqrt(math.factorial(i + 1)))

        total_f_i += curr_f / float(math.factorial(i + 1))
    return total_f_i

def get_spectrum(f_ab, frequencies, delta_e, max_e, anh_flag=False, gammas=[]):
    num_bins = int(np.ceil(max_e/delta_e))
    spectrum = np.zeros(num_bins, dtype=np.complex)
    if anh_flag:
        #print('using anharmonic gammas')
        for i in range(len(frequencies)):
            #print('i =', i)
            spectrum += f_ab[i] * create_lorentzian(frequencies[i], gammas[i], delta_e, max_e)
    else:
        for i in range(len(frequencies)):
            spectrum += f_ab[i] * create_delta(frequencies[i], delta_e, max_e)
    return spectrum

def get_atom_integers(masses):
    numbers = np.zeros(len(masses))
    curr_int = 0
    curr_mass = masses[0]
    for i, mass in enumerate(masses):
        if mass == curr_mass:
            numbers[i] = curr_int
        else:
            curr_int += 1
            curr_mass = mass
            numbers[i] = curr_int
    return numbers

def create_cell(lattice, positions, masses, magmoms=[]):
    numbers = get_atom_integers(masses)
    if len(magmoms) > 0:
        cell = (lattice, positions, numbers, magmoms)
    else:
        cell = (lattice, positions, numbers)
    return cell

def get_BZ_map(mesh, lattice, positions, masses, magmoms=[]):
    # what do I want????
    # I want to determine the BZ map, and full BZ grid
    # BZmap has shape of (N x 1) where N is the number of q-points in the full BZ
    bz_map = np.zeros(np.prod(mesh))
    numbers = get_atom_integers(masses)
    if len(magmoms) > 0:
        cell = (lattice, positions, numbers, magmoms)
    else:
        cell = (lattice, positions, numbers)
    bz_map, grid = spg.get_ir_reciprocal_mesh(mesh, cell)
    print(grid)
    grid = grid.astype(np.float)
    grid[:, 0] = grid[:, 0] / mesh[0]
    grid[:, 1] = grid[:, 1] / mesh[1]
    grid[:, 2] = grid[:, 2] / mesh[2]
    print(grid)
    return bz_map, grid

def get_symm_operations(lattice, positions, masses, magmoms=[]):
    numbers = get_atom_integers(masses)
    if len(magmoms) > 0:
        cell = (latice, positions, numbers, magmoms)
    else:
        cell = (lattice, positions, numbers)
    symm_dataset = spg.get_symmetry(cell)
    return (symm_dataset['rotations'], symm_dataset['translations'])

def get_equiv_atoms(lattice, positions, masses, magmoms=[]):
    cell = create_cell(lattice, positions, masses, magmoms)
    return spg.get_symmetry(cell)['equivalent_atoms']

def get_grid_index_from_address(address, mesh):
    grid_index = address[0] + mesh[0] * address[1] + mesh[0] * mesh[1] * address[2]
    return grid_index

def get_q_index_from_bzmap(address, bzmap, mesh):
    # address is a tuple of the form (i,j,k) where those indices are the indices for q-points in BZ
    # bzmap is a vector that contains integers that map the full BZ to the irreducible BZ
    # grid is a (Nqpts x 3) matrix containing the q-points in the full BZ in reduced coordinates.
    grid_index = get_grid_index_from_address(address, mesh)
    irr_grid_index = bzmap[grid_index]
    irr_grid_indices = np.unique(bzmap)
    q_index = np.where(irr_grid_index == irr_grid_indices)[0][0]
    return q_index

def get_eigindex_from_bzmap(address, bzmap, mesh, branch_num, total_branches):
    q_index = get_q_index_from_bzmap(address, bzmap, mesh)
    return q_index * total_branches + branch_num

def create_eigvec_grid(eigvecs, bzmap, mesh):
    #grid = np.zeros(mesh)
    #for i, g in enumerate(grid_points):
    #    grid[g[0], g[1], g[2]] = bzmap[i]
    natoms = eigvecs.shape[0]
    ncart = eigvecs.shape[1]
    neigs = eigvecs.shape[3]
    eig_list = np.array(
        [eigvecs[:, :, np.where(np.unique(bzmap) == g_id)[0][0], :] for g_id in bzmap])
    eig_grid = np.reshape(eig_list, np.concatenate((mesh, [natoms, ncart, neigs])))
    #for kx_id, g_yz in enumerate(grid):
    #    for ky_id, g_z in enumerate(g_yz):
    #        for kz_id, g in enumerate(g_y):
    #            eig_grid[:, :, :, kx_id, ky_id, kz_id] = eigvecs[:, :, grid[kx_id, ky_id, kz_id], :]
    return eig_grid

def create_qpt_map(bzmap, mesh):
    #bz_grid = np.reshape(bzmap, mesh)
    ir_map = np.unique(bzmap)
    #qpt_map = np.zeros(mesh)
    #for g_pt in bzmap:
    #    ind = np.where(ir_map == g_pt)[0][0]

    qpt_index_list = np.array([np.where(ir_map==g_pt)[0][0] for g_pt in bzmap])
    return np.reshape(qpt_index_list, mesh)

@jit(nopython=True, parallel=True)
def compute_Sqw(phonons, q_point, delta_e, max_e, num_overtones):
    # plan is to compute s_\tau\tau'(k, w) as presented in eq 22 in notes
    # once this function is constructed, then convolutions begin
    # BZ convolution performed using circ_conv (defined above) and frequency convolution done in normal (linear) way

    # assume q-point is in reduced coordinates
    if np.abs(q_point[0]) > 0.5 or np.abs(q_point[1]) > 0.5 or np.abs(q_point[2]) > 0.5:
        print("WARNING: q-point is outside the first Brillouin zone. Did you enter the q-point in reduced coordinates?")
    # convert reduced qpt to coordinates with real units (units = 2pi * m^-1)
    q_point = np.dot(q_point, phonons.rlattice) * 10**10

    phonons.unnormalize_eigvecs()
    num_bins = int(np.ceil(max_e / delta_e))
    #f_vec = np.zeros([phonons.natoms, num_bins], dtype=np.complex)
    s_fcn_size = np.append([phonons.natoms, phonons.natoms], phonons.mesh)
    s_fcn_size = np.append(s_fcn_size, num_bins)
    s_fcn = np.zeros(s_fcn_size, dtype=np.complex)

    # get BZ map
    print('mesh = ', phonons.mesh)
    print('lattice =', phonons.lattice)
    print('pos =', phonons.positions)
    print('masses =', phonons.masses)
    bz_map, grid = get_BZ_map(phonons.mesh, phonons.lattice, phonons.positions, phonons.masses)

    # calculate exp_DWF
    exp_DWF = np.zeros(phonons.natoms)
    for i in range(phonons.natoms):
        exp_DWF[i] = compute_f_DWF(q_point, phonons, i)
        #f_i = compute_f_i(q_point, phonons, i, delta_e, max_e)
        # don't convolve yet, just store into f_vec
        #f_vec[i, :] = f_i
    dist_matrix = np.zeros([phonons.natoms, phonons.natoms], dtype=np.complex)
    for i, pos1 in enumerate(phonons.positions):
        for j, pos2 in enumerate(phonons.positions):
            dist_matrix[i,j] = np.exp(np.vdot(1j * q_point, np.dot(pos2 - pos1, phonons.lattice)))

    print('total scattering at w=0:', sum(dist_matrix.flatten()))
    print('supercell =', phonons.supercell)
    #TODO: figuring out how normalization is related to num_cells variable
    #num_cells = np.abs(np.linalg.det(phonons.supercell))
    #norm_constant = 1 / phonons.natoms
    #norm_constant = 1 / (phonons.natoms * np.prod(phonons.mesh))
    norm_constant = 1 / (np.prod(phonons.mesh))
    num_cells = 1.0
    # loop over number of branches to add to s_fcn
    # this is where I construct s(k,w), which is a 4-d matrix
    n_eigs = 3 * phonons.natoms
    dxdydz_matrix = np.zeros([3, 3])
    for i in range(3):
        dxdydz_matrix[i, :] = phonons.rlattice[i, :] / phonons.mesh[i]
    dxdydz = np.abs(np.linalg.det(dxdydz_matrix))
    print('dxdydz =', dxdydz)
    dxdydzdw = dxdydz * delta_e
    print('dxdydzdw =', dxdydzdw)
    # NEW PLAN: make this object independent of q to spread with other MPI ranks
    # TODO: speed up code by changing how get_eigindex... code works; it currently uses np.where which appears to be unnecessary and slow
    # TODO: prob want to create code that maps the indices to a grid that makes sense, current mpgrid seems fine
    eig_grid = create_eigvec_grid(phonons.eigvecs, bz_map, phonons.mesh)
    qpt_map = create_qpt_map(bz_map, phonons.mesh)
    for tau_1 in range(phonons.natoms):
        for tau_2 in range(tau_1, phonons.natoms):
            for s in range(n_eigs):
                # loop over Brillouin zone grid to populate s_fcn
                for i in range(phonons.mesh[0]):
                    for j in range(phonons.mesh[1]):
                        for k in range(phonons.mesh[2]):
                            #print('tau1 = {}, tau2 = {}, s = {}, i = {}, j = {}, k = {}'.format(
                            #    tau_1, tau_2, s, i, j, k
                            #))
                            # need to convert grid position (i,j,k) to eigvec index in phonons.eigvecs
                            # use bz_map; maybe create a module
                            #eig_index = get_eigindex_from_bzmap((i,j,k), bz_map, phonons.mesh, s, n_eigs)

                            # hopefully eigenvectors are unnormalized at this point
                            # need to remove phonopy phase convention of eigvecs
                            r_1 = np.dot(phonons.positions[tau_1, :], phonons.lattice)
                            r_2 = np.dot(phonons.positions[tau_2, :], phonons.lattice)
                            #eig_1 = phonons.eigvecs[tau_1, :, eig_index] * np.exp(1j * np.vdot(q_point, r_1))
                            eig_1 = eig_grid[k, j, i, tau_1, :, s] * np.exp(1j * np.vdot(q_point, r_1))
                            #NOTE: in principle assuming that eig(k) = eig(-k)^* which may NOT be true in future
                            #      particularly when looking at potential topological materials
                            #eig_2 = phonons.eigvecs[tau_2, :, eig_index] * np.exp(1j * np.vdot(q_point, r_2))
                            eig_2 = eig_grid[k, j, i, tau_2, :, s] * np.exp(1j * np.vdot(q_point, r_2))
                            eig_index = qpt_map[k, j, i] * 3 * phonons.natoms + s
                            s_fcn[tau_1, tau_2, i, j, k, :] = norm_constant * (np.conj(np.vdot(q_point, eig_1)) * np.vdot(q_point, eig_2)) / \
                                                              num_cells * get_spectrum([dxdydzdw**-1],
                                                                                       [phonons.frequencies[eig_index]],
                                                                                       delta_e, max_e)
                            s_fcn[tau_2, tau_1, i, j, k, :] = np.conj(s_fcn[tau_1, tau_2, i, j, k, :])

    # now s_fcn is the proper 4-d object of tensors (6-d overall)
    #TODO note that the tensors can be completely decoupled and thus parallelized in future versions!!!
    s_qw = np.zeros(s_fcn.shape[2:], dtype=np.complex)
    # now compute full scattering function from convolutions of s_fcn
    norm_constant = 1

    for tau_1 in range(phonons.natoms):
        for tau_2 in range(phonons.natoms):
            r_1 = np.dot(phonons.positions[tau_1, :], phonons.lattice)
            r_2 = np.dot(phonons.positions[tau_2, :], phonons.lattice)
            s_qw += convolve_f_i(s_fcn[tau_1, tau_2, :, :, :, :], num_overtones, delta_e, coh_flag=True, dxdydz=dxdydz) * \
                    np.exp(1j * np.vdot(-q_point, (r_1 - r_2))) * exp_DWF[tau_1] * exp_DWF[tau_2]
    return s_qw

def compute_incoherent_Sqw(phonons, q_point, delta_e, max_e, num_overtones, anh_flag=False, gammas=[0]):
    # assume q-point is in reduced coordinates
    if np.abs(q_point[0]) > 0.5 or np.abs(q_point[1]) > 0.5 or np.abs(q_point[2]) > 0.5:
        print("WARNING: q-point is outside the first Brillouin zone. Did you enter the q-point in reduced coordinates?")
    # convert reduced qpt to coordinates with real units (units = 2pi * m^-1)
    q_point = np.dot(q_point, phonons.rlattice) * 10**10

    phonons.unnormalize_eigvecs()
    num_bins = int(np.ceil(max_e / delta_e))
    #f_vec = np.zeros([phonons.natoms, num_bins], dtype=np.complex)
    f_vec = np.zeros([phonons.natoms, len(phonons.frequencies)], dtype=np.complex)

    # calculate vector of partial scattering functions called f_i * exp_DWF
    exp_DWF = np.zeros(phonons.natoms)
    for i in range(phonons.natoms):
        exp_DWF[i] = compute_f_DWF(q_point, phonons, i)
        f_i = compute_f_i(q_point, phonons, i)
        # don't convolve yet, just store into f_vec
        f_vec[i, :] = f_i
    # now f_vec contains all information necessary to compute S_qw
    # need to multiply two different f_i's and convolve, then multiply by DWF's, and normalization constant (not implemented yet)
    s_qw = np.zeros(num_bins)
    # now compute full scattering function from pairs of partial scattering functions
    print('gammas =', gammas)
    for i in range(phonons.natoms):
        f_ab = np.abs(np.conj(f_vec[i, :]) * f_vec[i, :])
        print('sum of f_i = '), print(sum(f_ab))
        f_ab_spectrum = get_spectrum(f_ab, phonons.frequencies, delta_e, max_e, anh_flag, gammas)
        f_ab_spectrum = np.abs(f_ab_spectrum)
        print('int of f_i = '), print(np.trapz(f_ab_spectrum) * delta_e)
        f_ab_conv = convolve_f_i(f_ab_spectrum, num_overtones, delta_e)
        print('DWF')
        print(exp_DWF ** 2)
        s_qw += np.real(f_ab_conv) * exp_DWF[i] * exp_DWF[i]
    return s_qw

def compute_decoherence_time(phonons, q_point, delta_e, max_e, num_overtones, gammas, is_normalized=True):
    # assume q-point is in reduced coordinates
    if np.abs(q_point[0]) > 0.5 or np.abs(q_point[1]) > 0.5 or np.abs(q_point[2]) > 0.5:
        print("WARNING: q-point is outside the first Brillouin zone. Did you enter the q-point in reduced coordinates?")
    # convert reduced qpt to coordinates with real units (units = 2pi * m^-1)
    q_point = np.dot(q_point, phonons.rlattice) * 10**10
    if is_normalized:
        phonons.unnormalize_eigvecs()
    num_bins = int(np.ceil(max_e / delta_e))
    #f_vec = np.zeros([phonons.natoms, num_bins], dtype=np.complex)
    f_vec = np.zeros([phonons.natoms, len(phonons.frequencies)], dtype=np.complex)

    # calculate vector of partial scattering functions called f_i * exp_DWF
    exp_DWF = np.zeros(phonons.natoms)
    for i in range(phonons.natoms):
        exp_DWF[i] = compute_f_DWF(q_point, phonons, i)
        f_i = compute_f_i(q_point, phonons, i)
        # don't convolve yet, just store into f_vec
        f_vec[i, :] = f_i
    # now f_vec contains all information necessary to compute S_qw
    # need to multiply two different f_i's and convolve, then multiply by DWF's, and normalization constant (not implemented yet)
    s_qw = np.zeros(num_bins)
    gamma_s_qw = np.zeros(num_bins)
    # now compute full scattering function from pairs of partial scattering functions
    print('gammas =', gammas)
    for i in range(phonons.natoms):
        f_ab = np.abs(np.conj(f_vec[i, :]) * f_vec[i, :])
        gamma_f_ab = np.multiply(f_ab, np.exp(gammas))
        print('sum of f_i = '), print(sum(f_ab))
        f_ab_spectrum = get_spectrum(f_ab, phonons.frequencies, delta_e, max_e, True, gammas)
        f_ab_spectrum = np.abs(f_ab_spectrum)

        gamma_f_ab_spectrum = get_spectrum(gamma_f_ab, phonons.frequencies, delta_e, max_e, True, gammas)
        gamma_f_ab_spectrum = np.abs(gamma_f_ab_spectrum)
        print('int of f_i = '), print(np.trapz(f_ab_spectrum) * delta_e)
        f_ab_conv = convolve_f_i(f_ab_spectrum, num_overtones, delta_e)
        gamma_f_ab_conv = convolve_f_i(gamma_f_ab_spectrum, num_overtones, delta_e)
        print('DWF')
        print(exp_DWF ** 2)
        s_qw += np.real(f_ab_conv) * exp_DWF[i] * exp_DWF[i]
        gamma_s_qw += np.real(gamma_f_ab_conv) * exp_DWF[i] * exp_DWF[i]
    # need to take ln of each element in spectrum; cant use np.log because some elements in spectrum may be zero
    log_sqw = np.zeros(len(s_qw))
    log_gamma_sqw = np.zeros(len(gamma_s_qw))
    for i in range(len(log_sqw)):
        if s_qw[i] <= 0 or gamma_s_qw[i] <= 0:
            continue
        else:
            log_sqw[i] = np.log(s_qw[i])
            log_gamma_sqw[i] = np.log(gamma_s_qw[i])
    decoherence_time = 1 / (2 * np.trapz((log_gamma_sqw - log_sqw) * s_qw) / np.trapz(s_qw))
    return s_qw, decoherence_time;

def smear_spectrum(spectrum, sigma, delta_e):
    n_sigma = 10
    num_bins_half = int(np.ceil(n_sigma * sigma / delta_e))
    e_vals = np.arange(-num_bins_half, num_bins_half + 1) * delta_e
    gaussian = [1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-e**2 / (2 * sigma**2)) for e in e_vals]
    return signal.fftconvolve(spectrum, gaussian, mode='full')[num_bins_half:(num_bins_half + len(spectrum))] * delta_e

class Runner:
    def __init__(self, yaml_file, hdf5_file, q_point, num_overtones, max_e, delta_e):
        self.yaml_file = yaml_file
        self.hdf5_file = hdf5_file
        self.num_overtones = num_overtones
        self.max_e = max_e
        self.delta_e = delta_e
        self.q_point = q_point
    def get_coherent_spectrum(self):
        anh_phonons = phonon_lifetimes.Anh_System(self.yaml_file, self.hdf5_file)
        s_qw = compute_Sqw(anh_phonons.dyn_system, self.q_point, self.delta_e, self.max_e, self.num_overtones)
        mesh = anh_phonons.dyn_system.mesh
        q_bin = tuple(np.round((self.q_point % 1) * mesh).astype(np.int))
        print('int of s_qw for q =', self.q_point, 'is =', np.trapz(s_qw[q_bin]) * self.delta_e)
        return s_qw
    def get_phonons(self):
        return phonon_lifetimes.Anh_System(self.yaml_file, self.hdf5_file)

num_overtones = 1
delta_e = 0.01
# units are unfortunately in THz
max_e = 300

yaml_file = sys.argv[1]
hdf5_file = sys.argv[2]
#anh_phonons = phonon_lifetimes.Anh_System(yaml_file, hdf5_file)
#phonons = anh_phonons.dyn_system
#phonons = ph.Dyn_System(yaml_file)

#conv_THz_to_meV = 4.13567

# test with arbitrary single q-point
#reduced_qpt = 10 * np.array([0.5, 0.0, 0.0])
#d_times = np.zeros(3)

#mapping, testgrid = get_BZ_map(phonons.mesh, phonons.lattice, phonons.positions, phonons.masses, magmoms=[])

#s_qw = compute_Sqw(phonons, reduced_qpt, delta_e, max_e, num_overtones)
#s_qw = compute_incoherent_Sqw(phonons, reduced_qpt, delta_e, max_e, num_overtones)
#s_qw, decoherence_time = compute_decoherence_time(phonons, reduced_qpt, delta_e, max_e, num_overtones, anh_phonons.gammas, True)
"""#plot slices
ax = plt.gca()
ax.pcolormesh(np.squeeze(s_qw[:, 0, 0, :]))
plt.xlabel('qx')
plt.ylabel('omega')
plt.show()

plt.figure()
ax = plt.gca()
ax.pcolormesh(np.squeeze(s_qw[0, :, 0, :]))
plt.xlabel('qy')
plt.ylabel('omega')
plt.show()

plt.figure()
ax = plt.gca()
ax.pcolormesh(np.squeeze(s_qw[0, 0, :, :]))
plt.xlabel('qz')
plt.ylabel('omega')
plt.show()
"""

"""
for i in range(3):
#s_qw = compute_incoherent_Sqw(phonons, reduced_qpt, delta_e, max_e, num_overtones, True, anh_phonons.gammas)
    if i == 0:
        s_qw, decoherence_time = compute_decoherence_time(phonons, reduced_qpt, delta_e, max_e, num_overtones,
                                                          anh_phonons.gammas, True)
    else:
        s_qw, decoherence_time = compute_decoherence_time(phonons, reduced_qpt, delta_e, max_e, num_overtones,
                                                      anh_phonons.gammas, False)
    s_qw = smear_spectrum(s_qw, 0.5 / conv_THz_to_meV, delta_e)
    #g_s_qw = smear_spectrum(g_s_qw, 0.5 / conv_THz_to_meV, delta_e)
    print(s_qw)
    print('integral of s')
    print(np.trapz(s_qw) * delta_e)

    print('sum of weights = ')
    print(sum(phonons.weights))

    norm_constant = 1 #/ np.amax(s_qw)
    plt.plot(np.linspace(0, max_e, len(s_qw)) * conv_THz_to_meV, s_qw * norm_constant, label=('q = ' + str(reduced_qpt[0])))
    d_times[i] = decoherence_time
    reduced_qpt *= 100
    #plt.plot(np.linspace(0, max_e, len(s_qw)) * conv_THz_to_meV, g_s_qw)
#plt.xlim([15000, 20000])
plt.xlabel('Energy (meV)')
print('predicted average excitation energy = ')
print(const.hbar**2 * np.linalg.norm(np.dot(reduced_qpt, phonons.rlattice) * 10**10)**2 /
      (1.602*10**-19 * 2 * const.atomic_mass * np.average(phonons.masses)))
plt.xlim([0, 100])
plt.show()
plt.legend()
plt.yscale('log')

print('decoherence times =', d_times)
"""

