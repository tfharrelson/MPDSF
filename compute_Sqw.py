import numpy as np
from scipy import signal
import sys
#import yaml_phonons as ph
import phonon_lifetimes
import math
import scipy.constants as const
#import matplotlib.pyplot as plt
import h5py as h5
import spglib as spg
from phonopy import load
from phonopy.units import THz, AMU
from phonopy.interface.vasp import read_vasp
from phonopy.file_IO import parse_BORN
from phonopy.units import Bohr, Hartree
from phonopy.harmonic.force_constants import show_drift_force_constants
from phono3py.phonon3.fc3 import show_drift_fc3
from phono3py.api_phono3py import Phono3py
from phono3py.file_IO import (parse_disp_fc3_yaml,
                              parse_disp_fc2_yaml,
                              parse_FORCES_FC2,
                              parse_FORCES_FC3,
                              read_fc3_from_hdf5,
                              read_fc2_from_hdf5)
from spglib import get_ir_reciprocal_mesh
from scipy.interpolate import interp1d
AngstromsToMeters = 1e-10
#from numba import jit

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
class AnharmonicPhonons(object):
    def __init__(self,
                 poscar,
                 fc3_file,
                 disp_file,
                 mesh,
                 supercell,
                 temperature=0):
        self.cell = read_vasp(poscar)
        self.mesh = mesh
        self.phono3py = Phono3py(self.cell,
                                 supercell_matrix=supercell,
                                 primitive_matrix='auto',
                                 mesh=mesh,
                                 log_level=1)
        self.disp_data = parse_disp_fc3_yaml(filename=disp_file)
        self.fc3_data = parse_FORCES_FC3(self.disp_data, filename=fc3_file)
        self.phono3py.produce_fc3(self.fc3_data,
                                  displacement_dataset=self.disp_data,
                                  symmetrize_fc3r=True)
        ## initialize phonon-phonon interaction instance
        self.phono3py.init_phph_interaction()
        self.mapping = None
        self.grid = None
        self.irr_BZ_gridpoints = None
        self.phonon_freqs = None
        self.temperature = temperature
        #self.phono3py.run_imag_self_energy(np.unique(self.mapping), temperatures=temperature)

    def set_irr_BZ_gridpoints(self):
        self.mapping, grid = get_ir_reciprocal_mesh(mesh=mesh, cell=self.cell)
        self.grid = {tuple(k/mesh):v for (v, k) in enumerate(grid)}
        irr_BZ_gridpoints = np.unique(self.mapping)
        self.irr_BZ_gridpoints = {k:v for v, k in enumerate(irr_BZ_gridpoints)}
    def set_self_energies(self):
        #########################################################################################################
        # imaginary self energy objects are stored in phono3py object in a weird way                            #
        # the object is a list of a list of nparrays                                                            #
        # first index = grid-points, which are "addresses" of the irreducible q-points in the Brillouin zone    #
        #       At the surface these are meaningless, but the grid points are given by:                         #
        #       np.unique(mapping)                                                                              #
        #       The actual q-points are stored in grid with a 1-to-1 correspondence to mapping                  #
        # second index = sigma values, there is only one sigma=None in default tetrahedron method               #
        # third index is the nparray, which is arranged as:                                                     #
        #       (temperatures, frequency points, band index)                                                    #
        #########################################################################################################
        self.phono3py.init_phph_interaction()
        if self.mapping is None:
            self.set_irr_BZ_gridpoints()
        if type(self.temperature) is not list:
            self.temperature = [self.temperature]
        self.phono3py.run_imag_self_energy(np.unique(self.mapping), temperatures=self.temperature)
    def get_imag_self_energy(self, gridpoint, band_index):
        #########################################################################################################
        # Note that this returns the imaginary self energy function                                             #
        # Does not return the self energy evaluated at the phonon frequency                                     #
        # As a result, a tuple is returned with the frequency points, and self energy                           #
        #########################################################################################################
        irr_gp = self.mapping[gridpoint]
        grid_index = self.irr_BZ_gridpoints[irr_gp]
        return self.phono3py._frequency_points[grid_index][0], self.phono3py._imag_self_energy[grid_index][0][0, :, band_index]

    def get_imag_self_energy_interpolator(self, gridpoint, band_index):
        #########################################################################################################
        # This method will return an interpolator function                                                      #
        # The resulting function can be used as "y = ufunc(x)"                                                  #
        # The reason for creating an interpolator is because the frequency discretization is different in       #
        #       phono3py compared to my code below                                                              #
        #########################################################################################################
        freqs, ise = self.get_imag_self_energy(gridpoint, band_index)
        return interp1d(freqs, ise, kind='linear')

    def get_gridpoint(self, qpoint):

        for i, q in enumerate(qpoint):
            if q > 0.5:
                shift = np.ceil(q).astype(np.int)
                qpoint[i] = q - shift
            elif q < -0.5:
                shift = np.floor(q).astyp(np.int)
                qpoint[i] = q - shift
        # make sure qpoint is exactly a key
        key = np.round(np.array(qpoint) * np.array(self.mesh)).astype(int) / np.array(self.mesh)
        return self.grid[tuple(key)]

    def get_broadening_function(self, qpoint, band_index):
        #########################################################################################################
        # Method returns a broadening function that will resemble a simple Lorentzian                           #
        # However it is actually a Lorentzian-type function with a frequency-dependent width                    #
        # Since the frequency-dependent widths are still relatively small compared to the oscillator frequency  #
        #       the broadening function will look like a "normal" Lorentzian from afar                          #
        # One issue is that Lorentzians with widths smaller than the frequency spacing are not well defined     #
        #       which primarily means they are not normalized. This creates problems further down the pipeline  #
        # Normalize the integral by adding in a delta function with the remaining weight                        #
        #########################################################################################################
        self.set_phonon_freqs()
        gridpoint = self.get_gridpoint(qpoint)
        freqs, gamma = self.get_imag_self_energy(gridpoint, band_index)
        broadening_func = gamma / (np.pi * ((self.phonon_freqs[gridpoint, band_index] - freqs)**2 + gamma**2))
        f_index_minus = np.floor(self.phonon_freqs[gridpoint, band_index] / (freqs[1] - freqs[0])).astype(int)
        avg_gamma_at_freq = (gamma[f_index_minus] + gamma[f_index_minus+1]) / 2
        #print('original integral of broad func =', np.trapz(broadening_func, freqs))

        if np.trapz(broadening_func, freqs) < 1.0:
            if avg_gamma_at_freq < freqs[1]-freqs[0]:
                # Case: the width is small, and the area is underestimated
                # Solution: add a delta function
                broadening_func += self.create_delta(self.phonon_freqs[gridpoint, band_index],
                                                     len(freqs),
                                                     freqs[1]-freqs[0]) * \
                                   (1.0 - np.trapz(broadening_func, freqs))
            else:
                # Case: width is large enough, but the area is underestimated
                # Solution: scale the function by the trapz area
                broadening_func /= np.trapz(broadening_func, freqs)
        else:
            if avg_gamma_at_freq < freqs[1] - freqs[0]:
                # Case: width is too small to be resolved, area is overestimated by tails
                # Solution: subtract a delta function
                # Note: form is exactly the same as above because the (1-integral) term is now negative
                broadening_func += self.create_delta(self.phonon_freqs[gridpoint, band_index],
                                                     len(freqs),
                                                     freqs[1] - freqs[0]) * \
                                   (1.0 - np.trapz(broadening_func, freqs))
            else:
                # Case: width large enough to be resolved, but area is overestimated
                # Solution: scale by integral
                broadening_func /= np.trapz(broadening_func, freqs)
        # Notes: - can actually remove the first if statement as the Solution is always the same
        #        - Also, the function is a linear interpolation because the later convolution operations are compatible
        #          are discrete, and thus only compatible with linear interpolations.
        #        - For example, if we used a 'cubic' interpolator, the norm of the interpolated function is not
        #          guaranteed to be 1.0, which creates large problems when expanding to higher orders of perturbation.
        # Add large frequency mapped to zero for broad func to ensure no extrapolation errors

        #if np.trapz(broadening_func, freqs) != 0:
        #    broadening_func /= np.trapz(broadening_func, freqs)
        #print('adjusted integral of broad func =', np.trapz(broadening_func, freqs))
        freqs = np.append(freqs, freqs[-1]*1000)
        broadening_func = np.append(broadening_func, 0.0)
        #print('after extending, integral equals =', np.trapz(broadening_func, freqs))
        if gridpoint==0:
            #print('freq insided anharm code =', self.phonon_freqs[gridpoint, band_index])
            print('integral at gp0 =', np.trapz(broadening_func, freqs))
        return interp1d(freqs, broadening_func, kind='linear')

    def set_phonon_freqs(self):
        if self.phonon_freqs is None:
            self.phonon_freqs, _, _ = self.phono3py.get_phonon_data()
    def get_phonon_freqs(self):
        if self.phonon_freqs is None:
            self.set_phonon_freqs()
        return self.phonon_freqs
    def create_delta(self, energy, num_bins, delta_e):
        # units in meV
        delta_fcn = np.zeros(num_bins)

        if energy < 0:
            return delta_fcn

        e_bin_minus = int(np.floor(energy / delta_e))
        e_bin_plus = int(np.ceil(energy / delta_e))

        alpha_minus = np.abs(e_bin_minus * delta_e - energy) / delta_e
        alpha_plus = np.abs(e_bin_plus * delta_e - energy) / delta_e

        delta_fcn[e_bin_minus] = (1 - alpha_minus) / delta_e
        delta_fcn[e_bin_plus] = (1 - alpha_plus) / delta_e
        return delta_fcn

class DynamicStructureFactor(object):
    def __init__(self,
                 poscar_file,
                 fc_file,
                 mesh,
                 supercell,
                 q_point_list=[],
                 q_point_shift=[0.0, 0.0, 0.0],
                 fc3_file=None,
                 disp_file=None,
                 delta_e=0.01,
                 max_e=30,
                 num_overtones=10,
                 temperature=4,
                 freq_min=1e-3,
                 scattering_lengths=[],
                 primitive_flag='auto'):
        self.mesh = mesh
        self.supercell = supercell
        self.delta_e = delta_e
        self.max_e = max_e
        self.num_overtones = num_overtones
        self.temperature = temperature
        self.freq_min = freq_min
        if type(scattering_lengths) is not dict:
            #default atom type is Si
            self.scattering_lengths = {'Si':1.0}
        else:
            self.scattering_lengths = scattering_lengths

        # if qpoint list not given, then load from mesh and shift
        self.qpoint_shift = q_point_shift
        if len(q_point_list) == 0:
            self.qpoints = self.get_qpoint_list(self.mesh) + self.qpoint_shift
        else:
            self.qpoints = q_point_list
        self.kernel_qpoints = self.get_qpoint_list(self.mesh)
        if fc_file[-4:] == 'hdf5' or fc_file[-15:] == 'FORCE_CONSTANTS' or fc_file[-3:] == 'FC2':
            phonon = load(supercell_matrix=supercell,
                          primitive_matrix=primitive_flag,
                          unitcell_filename=poscar_file,
                          force_constants_filename=fc_file)
        elif fc_file[-10:] == 'FORCE_SETS':
            phonon = load(supercell_matrix=supercell,
                          primitive_matrix=primitive_flag,
                          unitcell_filename=poscar_file,
                          force_sets_filename=fc_file)
        else:
            print(fc_file, 'is not a recognized filetype!\nProgram exiting...')
            raise FileNotFoundError
        phonon.run_mesh(mesh,
                        is_mesh_symmetry=False,
                        with_eigenvectors=True)
        self.dsf = self.run_dsf(phonon, self.qpoints, self.temperature, scattering_lengths=self.scattering_lengths)
        self.sqw = []
        self.exp_DWF = []
        self.dxdydz = 0.0
        self.dxdydzdw = 0.0
        self.skw_kernel = []
        self.anharmonicities = None
        if fc3_file is not None and disp_file is not None:
            self.set_anharmonicities(poscar=poscar_file,
                                     fc3_file=fc3_file,
                                     disp_file=disp_file)

    def set_anharmonicities(self, poscar, fc3_file, disp_file):
        self.anharmonicities = AnharmonicPhonons(poscar=poscar,
                                                 fc3_file=fc3_file,
                                                 disp_file=disp_file,
                                                 mesh=self.mesh,
                                                 supercell=np.diag(self.supercell),
                                                 temperature=self.temperature
                                                 )
        self.anharmonicities.set_self_energies()

    def get_frequencies(self):
        num_bins = int(np.ceil(self.max_e / self.delta_e))
        return np.arange(num_bins) * self.delta_e

    def run_dsf(self,
            phonon,
            Qpoints,
            temperature,
            atomic_form_factor_func=None,
            scattering_lengths=None):
        # Transformation to the Q-points in reciprocal primitive basis vectors
        Q_prim = np.dot(Qpoints, phonon.primitive_matrix)
        # Q_prim must be passed to the phonopy dynamical structure factor code.
        phonon.run_dynamic_structure_factor(
            Q_prim,
            temperature,
            atomic_form_factor_func=atomic_form_factor_func,
            scattering_lengths=scattering_lengths,
            freq_min=1e-3)
        dsf = phonon.dynamic_structure_factor
        return dsf

    def get_qpoint_list(self, mesh):
        qpoints = np.zeros([np.prod(mesh), 3])
        count = 0
        curr_qx = 0.0
        curr_qy = 0.0
        curr_qz = 0.0
        spacing = 1.0 / np.array(mesh)
        for z in range(mesh[2]):
            for y in range(mesh[1]):
                for x in range(mesh[0]):
                    qpoints[count, :] = [curr_qx, curr_qy, curr_qz]
                    count += 1
                    curr_qx += spacing[0]
                    if curr_qx > 0.5:
                        curr_qx -= 1.0
                curr_qy += spacing[1]
                if curr_qy > 0.5:
                    curr_qy -= 1.0
            curr_qz += spacing[2]
            if curr_qz > 0.5:
                curr_qz -= 1.0
        return qpoints

    def get_outer_eig(self, eigvec, masses, freq, reduced_q, pos):
        outer_eig = np.zeros([len(masses), len(masses), 3, 3], dtype=np.complex)
        eigvec = eigvec.reshape(-1, 3)
        phase = np.exp(-2j * np.pi * np.dot(pos, reduced_q))
        for i in range(len(masses)):
            for j in range(i, len(masses)):
                outer_eig[i, j, :, :] = np.outer(np.conj(eigvec[i, :]), eigvec[j, :]) * const.hbar #* 2 * np.pi
                outer_eig[i, j, :, :] *= phase[i] * np.conj(phase[j]) / (2 * AMU * np.sqrt(masses[i] * masses[j])) / (2 * np.pi * freq * THz)

                if i is not j:
                    outer_eig[j, i, :, :] = np.conj(outer_eig[i, j, :, :])
        return outer_eig

    def get_outer_eigs_at_q(self, q_index):
        eigvecs = self.dsf._mesh_phonon.eigenvectors[q_index]
        masses = self.dsf._primitive.masses
        frequencies = self.dsf._mesh_phonon.frequencies[q_index]
        qpoint = self.kernel_qpoints[q_index]
        positions = self.dsf._primitive.get_scaled_positions()

        outer_eig_list = np.zeros([len(masses), len(masses), 3, 3, len(frequencies)], dtype=np.complex)
        for i, f in enumerate(frequencies):
            if self.dsf._fmin < f:
                outer_eig_list[:, :, :, :, i] = self.get_outer_eig(eigvecs[:, i], masses, f, qpoint, positions)
        return outer_eig_list

    def set_dxdydzdw(self):
        #dxdydz_matrix = np.empty([3, 3])
        #for i in range(3):
        #    dxdydz_matrix[i, :] = dsf._rec_lat[i, :] / mesh[i]
        #dxdydz = np.abs(np.linalg.det(dxdydz_matrix))
        #dxdydz = 1 / ((2 * np.pi)**3)
        dxdydz = 1.0
        print('dxdydz =', dxdydz)
        dxdydzdw = dxdydz * delta_e
        self.dxdydz = dxdydz
        self.dxdydzdw = dxdydzdw

    def build_skw_kernel(self):
        q_index = 0
        freqs = self.dsf._mesh_phonon.frequencies
        num_atoms = len(self.dsf._primitive.masses)
        num_bins = int(np.ceil(self.max_e / self.delta_e))
        skw_kernel = np.zeros(self.mesh + [num_atoms, num_atoms, 3, 3, num_bins], dtype=np.complex)
        norm_factor = 1 / np.prod(self.mesh)
        #norm_factor = 1
        if self.dxdydz == 0:
            self.set_dxdydzdw()

        for k in range(mesh[2]):
            for j in range(mesh[1]):
                for i in range(mesh[0]):
                    outer_eigs_at_q = self.get_outer_eigs_at_q(q_index)
                    freqs_at_q = freqs[q_index]

                    for outer_counter in range(outer_eigs_at_q.shape[-1]):
                        if q_index == 1:
                            print('band index =', outer_counter)
                            print('test outer_eigs =', outer_eigs_at_q[:, :, :, :, outer_counter])
                            print('freq in DSF code =', freqs_at_q[outer_counter])
                        if self.anharmonicities is None:
                            skw_kernel[i, j, k, :, :, :, :, :] += norm_factor * np.tensordot(
                                outer_eigs_at_q[:, :, :, :, outer_counter].reshape((1,) + outer_eigs_at_q.shape[:-1]),
                                self.get_spectrum([self.dxdydz ** -1], [freqs_at_q[outer_counter]]).reshape([1, num_bins]),
                                axes=[0, 0]
                            )
                        else:
                            skw_kernel[i, j, k, :, :, :, :, :] += norm_factor * np.tensordot(
                                outer_eigs_at_q[:, :, :, :, outer_counter].reshape((1,) + outer_eigs_at_q.shape[:-1]),
                                self.get_spectrum([self.dxdydz ** -1],
                                                  [freqs_at_q[outer_counter]],
                                                  q_point=self.kernel_qpoints[q_index],
                                                  band_index=outer_counter
                                                  ).reshape([1, num_bins]),
                                axes=[0, 0]
                            )
                    q_index += 1
        self.skw_kernel = skw_kernel

    def create_delta(self, energy):
        # units in meV
        num_bins = int(np.ceil(self.max_e / self.delta_e))
        delta_fcn = np.zeros(num_bins)

        if energy < 0:
            return delta_fcn

        e_bin_minus = int(np.floor(energy / self.delta_e))
        e_bin_plus = int(np.ceil(energy / self.delta_e))

        alpha_minus = np.abs(e_bin_minus * self.delta_e - energy) / self.delta_e
        alpha_plus = np.abs(e_bin_plus * self.delta_e - energy) / self.delta_e

        delta_fcn[e_bin_minus] = (1 - alpha_minus) / self.delta_e
        delta_fcn[e_bin_plus] = (1 - alpha_plus) / self.delta_e
        # if num_bins % 2 == 0:
        #    # even case
        #    delta_fcn[num_bins/2] = 1/(2 * delta_e)
        #    delta_fcn[num_bins/2 - 1] = 1/(2 * delta_e)
        # else:
        #    index = int(np.ceil(num_bins/2))
        #    delta_fcn[index] = 1 / delta_e
        return delta_fcn

    def create_anharmonic_distribution(self, q_point, band_index):
        freqs = self.get_frequencies()
        #print('qindex =', q_index)
        #print('kernel q =', self.kernel_qpoints[q_index].shape)
        #print('kernel q shape =', self.kernel_qpoints.shape)
        #gridpoint = self.anharmonicities.grid[tuple(self.kernel_qpoints[q_index, :])]
        #print('qpoint =', q_point)
        #print('band_index =', band_index)
        anh_dist_func = self.anharmonicities.get_broadening_function(q_point, band_index)
        return anh_dist_func(freqs)

    def create_lorentzian(self, energy, gamma):
        # create a Lorentzian function instead of a delta function for peaks broadened by anharmonicities
        # units of energy are THz for compatibility with phono3py

        num_bins = int(np.ceil(self.max_e / self.delta_e))
        lorentzian = np.zeros(num_bins)

        # check if energy is negative, return zero vector if true
        if energy < 0:
            return lorentzian

        # check if gamma is less than resolution of grid, if so return delta fcn instead of lorentzian
        if gamma < self.delta_e:
            return create_delta(energy, self.delta_e, self.max_e)

        # in an effort to speed up section
        n_sigma = 10
        # x_vals = np.arange(num_bins) * delta_e
        # lorentzian = np.array([(1 / np.pi * 0.5 * gamma / ((energy - x_val)**2 + (0.5 * gamma)**2))
        #                       if (x_val < energy + n_sigma * gamma) and (x_val > energy + n_sigma * gamma)
        #                       else 0.0 for x_val in x_vals])
        bin_spread = int(np.ceil(n_sigma * gamma / self.delta_e))
        e_bin = int(np.round(energy / self.delta_e))
        x_vals = np.arange(e_bin - bin_spread, e_bin + bin_spread + 1) * self.delta_e
        for i in range(len(x_vals)):
            lorentzian[i + e_bin - bin_spread] = 1 / np.pi * 0.5 * gamma / (
                        (energy - x_vals[i]) ** 2 + (0.5 * gamma) ** 2)
        return lorentzian

    def circ_conv(self, signal, kernel):
        # purpose is to perform a circular convolution (e.g. signal is periodic) of a 3-dimensional object
        # in this case, the 3-d object is a function proportional to phonon eigenvector as a function of q in the BZ
        # TODO Incorporate my own padding to the frequency dimension of the circular convolution; use appropriate tags
        return np.fft.ifftn(np.fft.fftn(signal) * np.fft.fftn(kernel))

    def convolve_f_i(self, f_i, coh_flag=False):
        # norm_constant = 1 / sum(phonons.weights)
        norm_constant = 1
        curr_f = f_i  # / sum(phonons.weights)
        f_i = f_i * norm_constant
        total_f_i = curr_f
        # norm_constant = 1 / 8
        # norm_constant = 1 / (8 * sum(phonons.weights))
        for i in range(1, self.num_overtones):
            print('overtone ='), print(i + 1)
            if i > 1:
                curr_f = curr_f * norm_constant
            # print('int of curr_f'), print(np.trapz(curr_f) * delta_e)
            if coh_flag:
                if i == 1:
                    curr_f *= 1
                curr_f = circ_conv(curr_f, f_i) * self.delta_e * self.dxdydz
            else:
                curr_f = signal.fftconvolve(curr_f, f_i, mode='full')[:len(f_i)] * self.delta_e
            # print('int of curr_f'), print(np.trapz(curr_f) * delta_e)
            # total_f_i += curr_f / (np.sqrt(math.factorial(i + 1)))

            total_f_i += curr_f / float(math.factorial(i + 1))
        return total_f_i

    def get_spectrum(self, f_ab, frequencies, q_point=None, band_index=None):
        num_bins = int(np.ceil(self.max_e / self.delta_e))
        spectrum = np.zeros(num_bins, dtype=np.complex)
        freqs = self.get_frequencies()
        if q_point is not None:
            # print('using anharmonic gammas')
            for i in range(len(frequencies)):
                #print('integral of broad func in DSF code =', np.trapz(self.create_anharmonic_distribution(q_point, band_index))*0.1)
                anh_dist = self.create_anharmonic_distribution(q_point, band_index)
                integral = np.trapz(anh_dist, freqs)
                if integral != 0:
                    spectrum += f_ab[i] * anh_dist / integral
        else:
            for i in range(len(frequencies)):
                spectrum += f_ab[i] * self.create_delta(frequencies[i])
        print('integral of spectrum =', np.trapz(spectrum, freqs))
        return spectrum

    def test_exp_DWF_at_q(self, q_index):
        eigvecs = self.dsf._mesh_phonon.eigenvectors
        weights = self.dsf._mesh_phonon.weights
        num_bands = eigvecs.shape[1]
        num_qpts = eigvecs.shape[0]

        q_cart = np.dot(self.qpoints[q_index], self.dsf._rec_lat) * (2 * np.pi / AngstromsToMeters)
        norm_constant = 1 / np.sum(weights)
        frequencies = self.dsf._mesh_phonon.frequencies
        masses = self.dsf._primitive.masses
        #eigvecs = np.reshape(eigvecs, [num_qpts, num_bands, len(masses), 3])
        exp_DWF = np.zeros(masses.shape)
        for m in range(len(masses)):
            DWF = 0.0
            for i, eigs_at_k in enumerate(eigvecs):
                for s, eig in enumerate(eigs_at_k.T):
                    if frequencies[i, s] > self.dsf._fmin:
                        eig = np.reshape(eig, [len(masses), 3])
                        DWF += np.abs(np.dot(q_cart, eig[m, :]))**2 * const.hbar / (4 * np.pi * frequencies[i, s] * THz * masses[m] * AMU)
            exp_DWF[m] = np.exp(-norm_constant * DWF/2)
        return exp_DWF

    def compute_exp_DWF_at_q(self, q_index):

        eigvecs = self.dsf._mesh_phonon.eigenvectors
        weights = self.dsf._mesh_phonon.weights
        num_bands = eigvecs.shape[1]
        num_qpts = eigvecs.shape[0]

        q_cart = np.dot(self.qpoints[q_index], self.dsf._rec_lat) * (2 * np.pi / AngstromsToMeters)
        norm_constant = 1 / np.sum(weights)
        frequencies = self.dsf._mesh_phonon.frequencies
        masses = self.dsf._primitive.masses
        eigvecs = np.reshape(eigvecs, [num_qpts, num_bands, len(masses), 3])
        # eigvecs now has shape (#qpts, #bands, #atoms, 3)
        # want to contract q with the cartesian indices, use np.dot
        q_dot_e = np.dot(eigvecs, q_cart)
        # find absolute value and square of all numbers in resulting matrix
        q_dot_e_sq = np.abs(q_dot_e)**2
        # multiply weights by norm_factor
        # multiply abs square by weights using transpose and np.multiply
        weighted_q_dot_e_sq = (weights * q_dot_e_sq.T * norm_constant * const.hbar).T
        weighted_q_dot_e_sq /= masses * AMU
        # remove negative frequencies
        inv_freqs = np.zeros(frequencies.shape)
        for i, freqs_at_q in enumerate(frequencies):
            for j, f in enumerate(freqs_at_q):
                if f > self.dsf._fmin:
                    inv_freqs[i, j] = 1 / (4 * np.pi * THz * f)
        weighted_q_dot_e_sq = ( np.reshape(inv_freqs, np.prod(frequencies.shape)) * \
                              np.reshape(weighted_q_dot_e_sq, [np.prod(frequencies.shape), len(masses)]).T).T

        return np.exp(-1 / 2 * np.sum(weighted_q_dot_e_sq, axis=0))

    def get_exp_DWF(self):
        for i in range(len(self.qpoints)):
            #self.exp_DWF.append(self.compute_exp_DWF_at_q(i))
            self.exp_DWF.append(self.test_exp_DWF_at_q(i))

    def get_coherent_sqw_at_q(self, q_index):
        if len(self.skw_kernel) is 0:
            self.build_skw_kernel()
        s_qw = np.zeros(self.skw_kernel.shape[:3] + (self.skw_kernel.shape[-1],), dtype=np.complex)
        # dot skw_kernel by q_point
        q_point = self.qpoints[q_index]
        q_cart = np.dot(q_point, self.dsf._rec_lat) * 2 * np.pi / AngstromsToMeters

        contracted_kernel = np.tensordot(self.skw_kernel, q_cart, axes=[[5], [0]])
        contracted_kernel = np.tensordot(contracted_kernel, q_cart, axes=[[5], [0]])
        #problem HERE with tensordot flipping the sign of the outerproducts
        # now compute full scattering function from convolutions of s_fcn
        natoms = len(self.dsf._primitive.masses)

        if len(self.exp_DWF) is 0:
            self.get_exp_DWF()

        norm_factor = self.dsf._unit_convertion_factor / (2 * np.pi * THz) * const.hbar
        positions = self.dsf._primitive.get_scaled_positions()
        for tau_1 in range(natoms):
            for tau_2 in range(natoms):
                s_qw[:, :, :, :] += self.convolve_f_i(contracted_kernel[:, :, :, tau_1, tau_2, :], coh_flag=True) * \
                        np.exp(2j * np.pi * np.vdot(q_point, (positions[tau_1] - positions[tau_2]))) \
                        * self.exp_DWF[q_index][tau_1] * self.exp_DWF[q_index][tau_2] #* norm_factor
        #TODO: Super worried about units, make sure they are correct, Togo's conversion factors seem wonky to me
        return s_qw

    def interpolate_sqw(self, sqw, q_point):
        #print('q-point =', q_point)
        #TODO save for later, implement actual interpolation scheme, right now will set up for the exact grid points
        indices = np.array(q_point * self.mesh - self.qpoint_shift).astype(np.int)
        indices = indices % self.mesh
        #print('indices =', indices)
        return sqw[indices[0], indices[1], indices[2], :]

    def get_coherent_sqw(self):
        for i in range(len(self.qpoints)):
            self.sqw.append(self.interpolate_sqw(self.get_coherent_sqw_at_q(i), self.qpoints[i]))
            print('i = ', i)
            print('integral of current sqw =', np.trapz(self.sqw[i])*0.1)

    def write_coherent_sqw(self, filename, ftype='hdf5'):
        if ftype == 'txt':
            fw = open(filename, 'w')
            fw.write('# first column is energy in THz, second column is the value of S(q, w)\n')
            for i, sqw_at_q in enumerate(self.sqw):
                fw.write('q-point = {}\n'.format(self.qpoints[i]))
                energy = 0.0
                for val in sqw_at_q:
                    fw.write('{:.3f}\t{:.10e}\n'.format(energy, val))
                    energy += self.delta_e
                fw.write('\n')
            fw.close()
        elif ftype == 'hdf5':

            with h5.File(filename, 'w') as fw:
                fw['q-points'] = self.qpoints
                fw['sqw'] = np.abs(np.array(self.sqw))
        else:
            print('ERROR: Unrecognized filetype used: ftype =', ftype)

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
        cell = (lattice, positions, numbers, magmoms)
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

def rearrange_eigvec(old_pos, new_pos, scrambled_eigvec):
    eigvec = np.zeros(scrambled_eigvec.shape, dtype=np.complex)
    for i, opos in enumerate(old_pos):
        for j, npos in enumerate(new_pos):
            if (npos == opos).all():
                eigvec[i, :] = scrambled_eigvec[j, :]
    return eigvec

def apply_symm_operations(eigvec, positions, q_pt, rots, trans):
    transformed_set = [eigvec]
    transformed_qs = [q_pt]
    print('transformed set =', transformed_set)
    for i, r in enumerate(rots):
        q_trans = np.dot(r, q_pt)
        if (q_pt == q_trans).all():
            continue
        else:
            q_flag = True
            for q_t in transformed_qs:
                if (q_t == q_trans).all():
                    q_flag = False
                    break
            if q_flag:
                transformed_qs.append(q_trans)
            else:
                continue
        scrambled_eigvec_rot = np.array([np.dot(r, eig) for eig in eigvec])

        #print('eigvec_rot =', eigvec_rot)

#        atom_trans = np.array([trans[i] for dummy in range(len(positions))]).T

        positions_rot = np.dot(positions, r)
        eigvec_rot = rearrange_eigvec(positions, positions_rot, scrambled_eigvec_rot)
        #NOTE: do not know whether the minus sign in the exponential is correct
        #NOTE: it comes from the translation operator, but phonopy may have its own phase convention that overrules the "standard" operation
        #eig_transform = np.empty(eigvec_rot.shape, dtype=np.complex)
        #exp_fcn = np.exp(-2j * np.pi * np.dot(q_pt, r_trans))
        #for count, er in enumerate(eigvec_rot):
        #    eig_transform[count, :] = er * exp_fcn[count]
        #transformed_set.append(eigvec_rot * np.exp(-1j * np.dot(q_pt, r_trans)))
        #transformed_set.append(eig_transform)
        transformed_set.append(eigvec_rot)
    return transformed_qs, transformed_set

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

def get_q_index(q_pt, mesh):
    return np.multiply(q_pt, mesh).astype(int)

#@jit(nopython=True, parallel=True)
def create_eigvec_grid_symm(q_pts, eigvecs, mesh, positions, lattice, masses, magmoms=[], checkweights=[]):
    """ This is an improvement of the module above called create_eigvec_grid. The purpose of this function is to create
    an eigenvector grid that correctly accounts for the symmetry operations present in the unit cell. The problem with
    the above module is that there are no symmetry operations applied, and no easy way to implement them"""
    (rots, trans) = get_symm_operations(lattice, positions, masses, magmoms)
    natoms = eigvecs.shape[0]
    ncart = eigvecs.shape[1]
    neigs = eigvecs.shape[3]
    print('neigs =', neigs)
    grid_shape = np.concatenate((mesh, [natoms, ncart, neigs]))
    eig_grid = np.empty(grid_shape, dtype=np.complex)
    for q_index in range(len(q_pts)):
        q = q_pts[q_index, :]
        print('q =', q)
        for eigvec_index in range(neigs):
            print('q_index =', q_index)
            eigvec = eigvecs[:, :, q_index, eigvec_index]
            (q_trans, eig_trans) = apply_symm_operations(eigvec, positions, q, rots, trans)
            print('weight check! weight from yaml file =', checkweights[q_index])
            print('weight from get_symm_operations in code =', len(q_trans))
            for i, q_t in enumerate(q_trans):
                q_indices = get_q_index(q_t, mesh)
                #print('eig_trans =', eig_trans[i][:, :])
                eig_grid[q_indices[2], q_indices[1], q_indices[0], :, :, eigvec_index] = eig_trans[i][:, :]
    return eig_grid

def create_qpt_map(bzmap, mesh):
    #bz_grid = np.reshape(bzmap, mesh)
    ir_map = np.unique(bzmap)
    #qpt_map = np.zeros(mesh)
    #for g_pt in bzmap:
    #    ind = np.where(ir_map == g_pt)[0][0]

    qpt_index_list = np.array([np.where(ir_map==g_pt)[0][0] for g_pt in bzmap])
    return np.reshape(qpt_index_list, mesh)

def get_qpt_shifts(mesh, gamma_center=True):
    #TODO: setup for non-gamma centered grids
    # Currently only set up for gamma center
    qpts = []
    for dim in mesh:
        spacing = 1.0 / dim
        curr_q = 0.0
        if not gamma_center:
            curr_q += spacing / 2
        qpts_along_dim = []
        for i in range(dim):
            if curr_q > 0.5:
                curr_q += -1
            qpts_along_dim.append(curr_q)
            curr_q += spacing
        qpts.append(np.array(qpts_along_dim))
    return np.array(qpts)

#@jit(nopython=True, parallel=True)
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
    #eig_grid = create_eigvec_grid_symm(phonons.eigvecs, bz_map, phonons.mesh)
    eig_grid = create_eigvec_grid_symm(phonons.qpoints, phonons.eigvecs, phonons.mesh, phonons.positions,
                                       phonons.lattice, phonons.masses, checkweights=phonons.weights)
    qpt_map = create_qpt_map(bz_map, phonons.mesh)
    #TODO: include gamma center tag in phonon object class
    qpt_shifts = get_qpt_shifts(phonons.mesh, gamma_center=True)
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
                            #r_1 = np.dot(phonons.positions[tau_1, :], phonons.lattice)
                            #r_2 = np.dot(phonons.positions[tau_2, :], phonons.lattice)
                            #eig_1 = phonons.eigvecs[tau_1, :, eig_index] * np.exp(1j * np.vdot(q_point, r_1))
                            # NOTE: Removing the phase factor convention because I think it's causing unintended problems
                            eig_1 = eig_grid[k, j, i, tau_1, :, s] * \
                                    np.exp(1j * np.vdot(np.array([qpt_shifts[0, i], qpt_shifts[1, j], qpt_shifts[2, k]]),
                                                        phonons.positions[tau_1, :]))
                            #NOTE: in principle assuming that eig(k) = eig(-k)^* which may NOT be true in future
                            #      particularly when looking at potential topological materials
                            #eig_2 = phonons.eigvecs[tau_2, :, eig_index] * np.exp(1j * np.vdot(q_point, r_2))
                            eig_2 = eig_grid[k, j, i, tau_2, :, s] * \
                                    np.exp(1j * np.vdot(np.array([qpt_shifts[0, i], qpt_shifts[1, j], qpt_shifts[2, k]]),
                                                        phonons.positions[tau_2, :]))
                            eig_index = qpt_map[k, j, i] * 3 * phonons.natoms + s
                            if i==5 and j==0 and k==0:
                                print('check dot prods...')
                                print('q dot eig2=', np.vdot(q_point, eig_1))
                                print('q dot eig1^* =', np.conj(np.vdot(q_point, eig_2)))
                                print('prod of both =', np.conj(np.vdot(q_point, eig_1)) * np.vdot(q_point, eig_2))
                                print('frequency =', phonons.frequencies[eig_index])
                            s_fcn[tau_1, tau_2, i, j, k, :] += norm_constant * (np.conj(np.vdot(q_point, eig_1)) * np.vdot(q_point, eig_2)) / \
                                                              num_cells * get_spectrum([dxdydzdw**-1],
                                                                                       [phonons.frequencies[eig_index]],
                                                                                       delta_e, max_e)
                            s_fcn[tau_2, tau_1, i, j, k, :] += np.conj(norm_constant * (np.conj(np.vdot(q_point, eig_1)) * np.vdot(q_point, eig_2)) / \
                                                              num_cells * get_spectrum([dxdydzdw**-1],
                                                                                       [phonons.frequencies[eig_index]],
                                                                                       delta_e, max_e))

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
                    np.exp(1j * np.vdot(q_point, (r_1 - r_2))) * exp_DWF[tau_1] * exp_DWF[tau_2]
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


if __name__ == '__main__':
    num_overtones = 2
    delta_e = 0.1
    # units are unfortunately in THz
    max_e = 100

    #yaml_file = sys.argv[1]
    #hdf5_file = sys.argv[2]
    #anh_phonons = phonon_lifetimes.Anh_System(yaml_file, hdf5_file)
    #phonons = anh_phonons.dyn_system
    #phonons = ph.Dyn_System(yaml_file)

    #conv_THz_to_meV = 4.13567

    # test with arbitrary single q-point
    reduced_qpt = np.outer(np.arange(1,46) / 9.0, np.array([1,0,0])) + 0

    print('reduced_qpoint =', reduced_qpt)
    #d_times = np.zeros(3)

    #mapping, testgrid = get_BZ_map(phonons.mesh, phonons.lattice, phonons.positions, phonons.masses, magmoms=[])
    mesh = [5,5,5]
    #supercell = [2, 2, 2]
    supercell = [1, 1, 1]
    #poscar = "/Volumes/GoogleDrive/My Drive/Cori_backup/GaAs/phono3py/2x2x2/POSCAR"
    poscar = "/Volumes/GoogleDrive/My Drive/multiphonon/rubrene_POSCAR"
    #fc = "/Volumes/GoogleDrive/My Drive/Cori_backup/GaAs/phono3py/2x2x2/fc2.hdf5"
    fc = "/Volumes/GoogleDrive/My Drive/multiphonon/rubrene_FORCE_SETS"
    #fc3 = "/Volumes/GoogleDrive/My Drive/Cori_backup/GaAs/phono3py/2x2x2/FORCES_FC3"
    #disp = "/Volumes/GoogleDrive/My Drive/Cori_backup/GaAs/phono3py/2x2x2/disp_fc3.yaml"
    scattering_lengths = {'C': 1.0, 'H': 1.0}
    #scattering_lengths = {'Ga': 1.0, 'As': 1.0}
    dynamic_structure_factor = DynamicStructureFactor(poscar, fc, mesh, supercell,
                                                      #q_point_list=reduced_qpt[0, :].reshape(-1, 3),
                                                      q_point_list=reduced_qpt,
                                                      max_e=max_e,
                                                      delta_e=delta_e,
                                                      num_overtones=num_overtones,
                                                      #fc3_file=fc3,
                                                      #disp_file=disp,
                                                      scattering_lengths=scattering_lengths)
    dynamic_structure_factor.get_coherent_sqw()
    dynamic_structure_factor.write_coherent_sqw('q0_anh_output_sqw.hdf5')
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

