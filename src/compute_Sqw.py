import numpy as np
from scipy import signal
# import yaml_phonons as ph
from src import phonon_lifetimes
import math
import scipy.constants as const
# import matplotlib.pyplot as plt
import h5py as h5
import spglib as spg
from phonopy import load, Phonopy
from phonopy.file_IO import parse_BORN
from phonopy.units import THz, AMU, Hartree, Bohr
from phonopy.interface.vasp import read_vasp
from phono3py.api_phono3py import Phono3py
from phono3py.file_IO import (parse_disp_fc3_yaml,
                              parse_FORCES_FC3)
from spglib import get_ir_reciprocal_mesh
from scipy.interpolate import interp1d

AngstromsToMeters = 1e-10
# from numba import jit

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
                                 supercell_matrix=np.diag(supercell),
                                 primitive_matrix='auto',
                                 mesh=mesh,
                                 log_level=1)
        self.disp_data = parse_disp_fc3_yaml(filename=disp_file)
        self.fc3_data = parse_FORCES_FC3(self.disp_data, filename=fc3_file)
        self.phono3py.produce_fc3(self.fc3_data,
                                  displacement_dataset=self.disp_data,
                                  symmetrize_fc3r=True)
        #TODO: Add BORN conditional statement here
        ## initialize phonon-phonon interaction instance
        self.phono3py.init_phph_interaction()
        self.mapping = None
        self.grid = None
        self.irr_BZ_gridpoints = None
        self.phonon_freqs = None
        self.temperature = temperature
        # self.phono3py.run_imag_self_energy(np.unique(self.mapping), temperatures=temperature)

    def set_irr_BZ_gridpoints(self):
        self.mapping, grid = get_ir_reciprocal_mesh(mesh=self.mesh, cell=self.cell)
        self.grid = {tuple(k / self.mesh): v for (v, k) in enumerate(grid)}
        irr_BZ_gridpoints = np.unique(self.mapping)
        self.irr_BZ_gridpoints = {k: v for v, k in enumerate(irr_BZ_gridpoints)}

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
        return self.phono3py._frequency_points, self.phono3py._gammas[grid_index][0][0, band_index, :]

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
                shift = np.floor(q).astype(np.int)
                qpoint[i] = q - shift
        # make sure qpoint is exactly a key
        key = np.round(np.array(qpoint) * np.array(self.mesh)).astype(int) / np.array(self.mesh)
        return self.grid[tuple(key)]

    def get_broadening_function(self, qpoint, band_index, max_freq=None):
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
        broadening_func = gamma / (np.pi * ((self.phonon_freqs[gridpoint, band_index] - freqs) ** 2 + gamma ** 2))
        f_index_minus = np.floor(self.phonon_freqs[gridpoint, band_index] / (freqs[1] - freqs[0])).astype(int)
        avg_gamma_at_freq = (gamma[f_index_minus] + gamma[f_index_minus + 1]) / 2
        # print('original integral of broad func =', np.trapz(broadening_func, freqs))

        if max_freq is None:
            freqs = np.append(freqs, freqs[-1] * 1000)
        else:
            freqs = np.append(freqs, max_freq + 1.0)
        broadening_func = np.append(broadening_func, 0.0)

        if np.trapz(broadening_func, freqs) < 1.0:
            if avg_gamma_at_freq < freqs[1] - freqs[0]:
                # Case: the width is small, and the area is underestimated
                # Solution: add a delta function
                broadening_func += self.create_delta(self.phonon_freqs[gridpoint, band_index],
                                                     len(freqs),
                                                     freqs[1] - freqs[0]) * \
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

        # increase interpolation range by adding zeros out to maximum frequency
        # if no maximum frequency specified, then multiply last frequency by 1000
        # print('after extending, integral equals =', np.trapz(broadening_func, freqs))
        # if gridpoint==0:
        # print('freq insided anharm code =', self.phonon_freqs[gridpoint, band_index])
        # print('integral at gp0 =', np.trapz(broadening_func, freqs))
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

    def set_gamma_interpolator(self):
        from src.Interpolation import Interpolator

        gamma_data, qpts, freqs = self.get_gamma_data()
        interp = Interpolator(gamma=gamma_data, qpoints=qpts, freqs=freqs)

    def get_gamma_data(self, band_index):
        gamma_data = []
        q_pts = []
        for q in self.phono3py.get_phonon_data()[2]:
            q /= self.phono3py.mesh_numbers
            freqs, gamma = self.get_imag_self_energy(self.get_gridpoint(q), band_index)
            # gamma_data += list(gamma)
            # q_data += [list(q) + [freq] for freq in freqs]
            q_pts.append(q)
            gamma_data.append(gamma)
        return gamma_data, q_pts, freqs


class DynamicStructureFactor(object):
    def __init__(self,
                 poscar_file,
                 mesh,
                 supercell,
                 fc_file=None,
                 q_point_list=[],
                 q_point_shift=[0.0, 0.0, 0.0],
                 fc2_disp=None,
                 fc3_file=None,
                 fc3_disp=None,
                 delta_e=0.01,
                 max_e=30,
                 num_overtones=10,
                 temperature=4,
                 freq_min=1e-3,
                 primitive_flag='auto',
                 is_nac=False,
                 born_file=None,
                 scalar_mediator_flag=True,
                 dark_photon_flag=False):
        self.mesh = mesh
        self.supercell = supercell
        self.delta_e = delta_e
        self.max_e = max_e
        self.num_overtones = num_overtones
        self.temperature = temperature
        self.freq_min = freq_min
        self.is_nac = is_nac

        self.disp_data = parse_disp_fc3_yaml(filename=fc3_disp)
        self.fc3_data = parse_FORCES_FC3(self.disp_data, filename=fc3_file)
        phono3py = Phono3py(read_vasp(poscar_file),
                            supercell_matrix=np.diag(supercell),
                            primitive_matrix='auto',
                            mesh=mesh,
                            log_level=1)
        phono3py.produce_fc3(self.fc3_data, self.disp_data)

        # if qpoint list not given, then load from mesh and shift
        self.qpoint_shift = q_point_shift
        if len(q_point_list) == 0:
            self.qpoints = self.get_qpoint_list(self.mesh)[1:] + self.qpoint_shift
        else:
            self.qpoints = q_point_list
        #self.kernel_qpoints = self.get_qpoint_list(self.mesh)
        if fc_file is not None:
            if fc_file[-4:] == 'hdf5' or fc_file[-15:] == 'FORCE_CONSTANTS':
                phonon = load(supercell_matrix=supercell,
                              primitive_matrix=primitive_flag,
                              unitcell_filename=poscar_file,
                              force_constants_filename=fc_file,
                              is_nac=is_nac,
                              born_filename=born_file)
            elif fc_file[-10:] == 'FORCE_SETS' or fc_file[-3:] == 'FC2' or fc_file is None:
                phonon = load(supercell_matrix=supercell,
                              primitive_matrix=primitive_flag,
                              unitcell_filename=poscar_file,
                              force_sets_filename=fc_file,
                              is_nac=is_nac,
                              born_filename=born_file)
            else:
                print(fc_file, 'is not a recognized filetype!\nProgram exiting...')
                raise FileNotFoundError
        else:
                nac_params = parse_BORN(phono3py.get_phonon_primitive(), filename=born_file)
                nac_params['factor'] = Hartree * Bohr
                phonon = Phonopy(read_vasp(poscar_file),
                                 supercell_matrix=supercell,
                                 primitive_matrix=primitive_flag,
                                 nac_params=nac_params)

                fc2 = phono3py.get_fc2()
                phonon.force_constants = fc2

        self.phonon = phonon
        self.eigenvectors = None
        self.frequencies = None
        self.weights = None
        self.kernel_qpoints = None
        self.run_mesh()
        self.rec_lat = np.linalg.inv(self.phonon.primitive.get_cell())

        self.sqw = []
        self.exp_DWF = []
        self.dxdydz = 0.0
        self.dxdydzdw = 0.0
        self.skw_kernel = []
        self.anharmonicities = None
        self.born_charges = None
        self.dielectric = None
        if self.is_nac is True or born_file is not None:
            self.set_born_charges()
            self.set_dielectric()
        self._scalar_mediator_flag = scalar_mediator_flag
        self._dark_photon_flag = dark_photon_flag
        if fc3_file is not None and fc3_disp is not None:
            self.set_anharmonicities(poscar=poscar_file,
                                     fc3_file=fc3_file,
                                     disp_file=fc3_disp)

    def run_mesh(self):
        self.phonon.run_mesh(self.mesh,
                             shift=self.qpoint_shift,
                             is_mesh_symmetry=False,
                             with_eigenvectors=True,
                             is_gamma_center=True)
        mesh_dict = self.phonon.get_mesh_dict()
        self.eigenvectors = mesh_dict['eigenvectors']
        self.frequencies = mesh_dict['frequencies']
        self.weights = mesh_dict['weights']
        self.kernel_qpoints = mesh_dict['qpoints']

    def set_born_charges(self):
        self.born_charges = self.phonon.nac_params['born']

    def set_dielectric(self):
        self.dielectric = self.phonon.nac_params['dielectric']

    def set_anharmonicities(self, poscar, fc3_file, disp_file):
        self.anharmonicities = AnharmonicPhonons(poscar=poscar,
                                                 fc3_file=fc3_file,
                                                 disp_file=disp_file,
                                                 mesh=self.mesh,
                                                 supercell=self.supercell,
                                                 temperature=self.temperature
                                                 )
        self.anharmonicities.set_self_energies()

    def get_bin_energies(self):
        num_bins = int(np.ceil(self.max_e / self.delta_e))
        return np.arange(num_bins) * self.delta_e

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
                if self._dark_photon_flag is True:
                    outer_eig[i, j, :, :] = np.outer(np.conj(np.dot(eigvec[i, :], self.born_charges[i].T)),
                                                     np.dot(self.born_charges[j], eigvec[j, :].T)) * const.hbar
                else:
                    outer_eig[i, j, :, :] = np.outer(np.conj(eigvec[i, :]), eigvec[j, :]) * const.hbar
                outer_eig[i, j, :, :] *= phase[i] * np.conj(phase[j]) / (2 * AMU * np.sqrt(masses[i] * masses[j])) / (
                        2 * np.pi * freq * THz)

                if i is not j:
                    outer_eig[j, i, :, :] = np.conj(outer_eig[i, j, :, :])
        return outer_eig

    def get_outer_eigs_at_q(self, q_index):
        eigvecs = self.eigenvectors[q_index]
        masses = self.phonon.masses
        frequencies = self.frequencies[q_index]
        qpoint = self.kernel_qpoints[q_index]
        positions = self.phonon.primitive.get_scaled_positions()

        outer_eig_list = np.zeros([len(masses), len(masses), 3, 3, len(frequencies)], dtype=np.complex)
        for i, f in enumerate(frequencies):
            if q_index == 0 and i < 3:
                continue
            if self.freq_min < f:
                outer_eig_list[:, :, :, :, i] = self.get_outer_eig(eigvecs[:, i], masses, f, qpoint, positions)
        return outer_eig_list

    def set_dxdydzdw(self):
        dxdydz_matrix = np.empty([3, 3])
        for i in range(3):
            dxdydz_matrix[i, :] = 2 * np.pi * self.rec_lat[i, :] / self.mesh[i]
        dxdydz = np.abs(np.linalg.det(dxdydz_matrix))
        dxdydzdw = dxdydz * self.delta_e
        self.dxdydz = dxdydz
        self.dxdydzdw = dxdydzdw

    def build_skw_kernel(self):
        q_index = 0
        freqs = self.frequencies
        num_atoms = len(self.phonon.primitive.masses)
        num_bins = int(np.ceil(self.max_e / self.delta_e))
        skw_kernel = np.zeros(self.mesh + [num_atoms, num_atoms, 3, 3, num_bins], dtype=np.complex)
        norm_factor = 1 / np.prod(self.mesh)
        # norm_factor = 1
        if self.dxdydz == 0:
            self.set_dxdydzdw()

        for k in range(self.mesh[2]):
            for j in range(self.mesh[1]):
                for i in range(self.mesh[0]):
                    outer_eigs_at_q = self.get_outer_eigs_at_q(q_index)
                    freqs_at_q = freqs[q_index]

                    for outer_counter in range(outer_eigs_at_q.shape[-1]):
                        if self.anharmonicities is None:
                            skw_kernel[i, j, k, :, :, :, :, :] += norm_factor * np.tensordot(
                                outer_eigs_at_q[:, :, :, :, outer_counter].reshape((1,) + outer_eigs_at_q.shape[:-1]),
                                self.get_spectrum([self.dxdydz ** -1], [freqs_at_q[outer_counter]]).reshape(
                                    [1, num_bins]),
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
        freqs = self.get_bin_energies()
        # print('qindex =', q_index)
        # print('kernel q =', self.kernel_qpoints[q_index].shape)
        # print('kernel q shape =', self.kernel_qpoints.shape)
        # gridpoint = self.anharmonicities.grid[tuple(self.kernel_qpoints[q_index, :])]
        # print('qpoint =', q_point)
        # print('band_index =', band_index)
        anh_dist_func = self.anharmonicities.get_broadening_function(q_point, band_index, max_freq=self.max_e)
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
            return self.create_delta(energy)

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
            # print('overtone ='), print(i + 1)
            if i > 1:
                curr_f = curr_f * norm_constant
            # print('int of curr_f'), print(np.trapz(curr_f[1,0,0,:]) * self.delta_e)
            if coh_flag:
                if i == 1:
                    curr_f *= 1
                curr_f = self.circ_conv(curr_f, f_i) * self.delta_e * self.dxdydz
            else:
                curr_f = signal.fftconvolve(curr_f, f_i, mode='full')[:len(f_i)] * self.delta_e
            # print('int of curr_f'), print(np.trapz(curr_f[1,0,0,:]) * self.delta_e)
            # total_f_i += curr_f / (np.sqrt(math.factorial(i + 1)))

            total_f_i += curr_f / float(math.factorial(i + 1))
        return total_f_i

    def get_spectrum(self, f_ab, frequencies, q_point=None, band_index=None):
        num_bins = int(np.ceil(self.max_e / self.delta_e))
        spectrum = np.zeros(num_bins, dtype=np.complex)
        freqs = self.get_bin_energies()
        if q_point is not None:
            # print('using anharmonic gammas')
            for i in range(len(frequencies)):
                # print('integral of broad func in DSF code =', np.trapz(self.create_anharmonic_distribution(q_point, band_index))*0.1)
                anh_dist = self.create_anharmonic_distribution(q_point, band_index)
                integral = np.trapz(anh_dist, freqs)
                # print('integral =', integral)
                if integral != 0:
                    spectrum += f_ab[i] * anh_dist / integral
        else:
            for i in range(len(frequencies)):
                spectrum += f_ab[i] * self.create_delta(frequencies[i])
        # print('integral of spectrum =', np.trapz(spectrum, freqs))
        return spectrum

    def test_exp_DWF_at_q(self, q_index):
        eigvecs = self.eigenvectors
        weights = self.weights
        num_bands = eigvecs.shape[1]
        num_qpts = eigvecs.shape[0]

        q_cart = np.dot(self.qpoints[q_index], self.rec_lat) * (2 * np.pi / AngstromsToMeters)
        norm_constant = 1 / np.sum(weights)
        frequencies = self.frequencies
        masses = self.phonon.primitive.masses
        # eigvecs = np.reshape(eigvecs, [num_qpts, num_bands, len(masses), 3])
        exp_DWF = np.zeros(masses.shape)
        for m in range(len(masses)):
            DWF = 0.0
            for i, eigs_at_k in enumerate(eigvecs):
                for s, eig in enumerate(eigs_at_k.T):
                    if i == 0 and s < 3:
                        continue
                    if frequencies[i, s] > self.freq_min:
                        eig = np.reshape(eig, [len(masses), 3])
                        DWF += np.abs(np.dot(q_cart, eig[m, :])) ** 2 * const.hbar / (
                                4 * np.pi * frequencies[i, s] * THz * masses[m] * AMU)
            exp_DWF[m] = np.exp(-norm_constant * DWF / 2)
        return exp_DWF

    def compute_exp_DWF_at_q(self, q_index):

        eigvecs = self.eigenvectors
        weights = self.weights
        num_bands = eigvecs.shape[1]
        num_qpts = eigvecs.shape[0]

        q_cart = np.dot(self.qpoints[q_index], self.rec_lat) * (2 * np.pi / AngstromsToMeters)
        norm_constant = 1 / np.sum(weights)
        frequencies = self.frequencies
        masses = self.phonon.primitive.masses
        eigvecs = np.reshape(eigvecs, [num_qpts, num_bands, len(masses), 3])
        # eigvecs now has shape (#qpts, #bands, #atoms, 3)
        # want to contract q with the cartesian indices, use np.dot
        q_dot_e = np.dot(eigvecs, q_cart)
        # find absolute value and square of all numbers in resulting matrix
        q_dot_e_sq = np.abs(q_dot_e) ** 2
        # multiply weights by norm_factor
        # multiply abs square by weights using transpose and np.multiply
        weighted_q_dot_e_sq = (weights * q_dot_e_sq.T * norm_constant * const.hbar).T
        weighted_q_dot_e_sq /= masses * AMU
        # remove negative frequencies
        inv_freqs = np.zeros(frequencies.shape)
        for i, freqs_at_q in enumerate(frequencies):
            for j, f in enumerate(freqs_at_q):
                if f > self.freq_min:
                    inv_freqs[i, j] = 1 / (4 * np.pi * THz * f)  # Represents 1/2w, where w is the angular frequency
        weighted_q_dot_e_sq = (np.reshape(inv_freqs, np.prod(frequencies.shape)) * \
                               np.reshape(weighted_q_dot_e_sq, [np.prod(frequencies.shape), len(masses)]).T).T

        return np.exp(-1 / 2 * np.sum(weighted_q_dot_e_sq, axis=0))

    def get_exp_DWF(self):
        for i in range(len(self.qpoints)):
            # self.exp_DWF.append(self.compute_exp_DWF_at_q(i))
            self.exp_DWF.append(self.test_exp_DWF_at_q(i))

    def get_coherent_sqw_at_q(self, q_index):
        if len(self.skw_kernel) is 0:
            self.build_skw_kernel()
        s_qw = np.zeros(self.skw_kernel.shape[:3] + (self.skw_kernel.shape[-1],), dtype=np.complex)
        # dot skw_kernel by q_point
        q_point = self.qpoints[q_index]
        q_cart = np.dot(q_point, self.rec_lat) * 2 * np.pi / AngstromsToMeters

        masses = self.phonon.primitive.masses
        natoms = len(masses)

        contracted_kernel = np.tensordot(self.skw_kernel, q_cart, axes=[[5], [0]])
        contracted_kernel = np.tensordot(contracted_kernel, q_cart, axes=[[5], [0]])
        # problem HERE with tensordot flipping the sign of the outerproducts
        # now compute full scattering function from convolutions of s_fcn

        if len(self.exp_DWF) is 0:
            self.get_exp_DWF()

        norm_factor = np.prod(self.mesh)
        positions = self.phonon.primitive.get_scaled_positions()
        for tau_1 in range(natoms):
            for tau_2 in range(natoms):
                factor = self.convolve_f_i(contracted_kernel[:, :, :, tau_1, tau_2, :], coh_flag=True) * \
                         np.exp(2j * np.pi * np.vdot(q_point, (positions[tau_1] - positions[tau_2]))) \
                         * self.exp_DWF[q_index][tau_1] * self.exp_DWF[q_index][tau_2] * norm_factor
                if self._scalar_mediator_flag is True:
                    factor *= masses[tau_1] * masses[tau_2]
                elif self._dark_photon_flag is True:
                    # TODO: Consider moving out of all loops as this factor only depends on q, not loop indices
                    factor *= np.abs(np.linalg.norm(q_cart) ** 2 /
                                     (np.dot(q_cart, np.dot(self.dielectric, q_cart)))
                                     ) ** 2

                s_qw[:, :, :, :] += factor
        # TODO: Super worried about units, make sure they are correct, Togo's conversion factors seem wonky to me
        return s_qw

    def get_indices_from_qpoint(self, q_point):
        indices = np.round(q_point * self.mesh - self.qpoint_shift).astype(np.int)
        indices = indices % self.mesh
        return indices

    def interpolate_sqw(self, sqw, q_point):
        # print('q-point =', q_point)
        # TODO save for later, implement actual interpolation scheme, right now will set up for the exact grid points
        indices = np.array(q_point * self.mesh - self.qpoint_shift).astype(np.int)
        indices = indices % self.mesh
        # print('indices =', indices)
        return sqw[indices[0], indices[1], indices[2], :]

    def get_coherent_sqw(self, start_q_index=None, stop_q_index=None):
        if start_q_index is None:
            start_q_index = 0
        if stop_q_index is None:
            stop_q_index = len(self.qpoints)
        print('testing qpoints =', self.qpoints)
        for i in range(start_q_index, stop_q_index):
            if np.linalg.norm(self.qpoints[i]) == 0.:
                if len(self.skw_kernel) is 0:
                    self.build_skw_kernel()
                self.sqw.append(np.zeros(self.skw_kernel.shape[:3] + (self.skw_kernel.shape[-1],), dtype=np.complex))
            else:
                self.sqw.append(self.interpolate_sqw(self.get_coherent_sqw_at_q(i), self.qpoints[i]))
            print('i = ', i)
#            print('integral of current sqw =', np.trapz(self.sqw[i]) * self.dxdydzdw)

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
                fw['reclat'] = self.rec_lat * 2 * np.pi
                fw['frequencies'] = self.get_bin_energies()
                fw['delta_w'] = self.delta_e
                fw['dxdydz'] = self.dxdydz
        else:
            print('ERROR: Unrecognized filetype used: ftype =', ftype)


'''
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
'''


def set_qpoints(mesh, num_Gpoints=1, q_shift=[0, 0, 0], stride=1, stride_G=1, max_q=None):
    qpoints = np.zeros([0, 3])
    count = 0
    curr_qx = q_shift[0] / mesh[0]
    curr_qy = q_shift[1] / mesh[1]
    curr_qz = q_shift[2] / mesh[2]
    spacing = float(stride) / np.array(mesh)
    for gz in range(0, num_Gpoints, stride_G):
        for gy in range(0, num_Gpoints, stride_G):
            for gx in range(0, num_Gpoints, stride_G):
                for z in range(0, mesh[2], stride):
                    for y in range(0, mesh[1], stride):
                        for x in range(0, mesh[0], stride):
                            if max_q is not None:
                                if np.abs(curr_qx) <= max_q and np.abs(curr_qy) <= max_q and np.abs(curr_qz) <= max_q:
                                    qpoints = np.append(qpoints, [[curr_qx + gx, curr_qy + gy, curr_qz + gz]], axis=0)
                            else:
                                qpoints = np.append(qpoints, [[curr_qx + gx, curr_qy + gy, curr_qz + gz]], axis=0)
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
        print('gz =', gz)
    return qpoints


if __name__ == '__main__':
    num_overtones = 1
    delta_e = 0.1
    # units are unfortunately in THz
    max_e = 100

    # yaml_file = sys.argv[1]
    # hdf5_file = sys.argv[2]
    # anh_phonons = phonon_lifetimes.Anh_System(yaml_file, hdf5_file)
    # phonons = anh_phonons.dyn_system
    # phonons = ph.Dyn_System(yaml_file)

    # conv_THz_to_meV = 4.13567

    # test with arbitrary single q-point
    reduced_qpt = np.outer(np.arange(1, 12) / 23.0, np.array([1, 0, 0])) + 0
    reduced_qpts = reduced_qpt
    reduced_qpts = np.append(reduced_qpts, np.outer(np.arange(3, 4) / 5.0, np.array([1, 1, 1])) + 10, axis=0)
    reduced_qpts = np.append(reduced_qpts, np.outer(np.arange(3, 4) / 5.0, np.array([1, 1, 1])) + 20, axis=0)
    reduced_qpts = np.append(reduced_qpts, np.outer(np.arange(3, 4) / 5.0, np.array([1, 1, 1])) + 50, axis=0)
    # test with entire BZ

    # print(qpoints)
    # mapping, testgrid = get_BZ_map(phonons.mesh, phonons.lattice, phonons.positions, phonons.masses, magmoms=[])
    mesh = [9, 9, 9]
    supercell = [2, 2, 2]

    qpoints = set_qpoints(mesh=mesh)  # , num_Gpoints=10)#, num_Gpoints=300, stride_G=50)
    print(qpoints[:10, :])
    # supercell = [1, 1, 1]
    poscar = "/Volumes/GoogleDrive/My Drive/Cori_backup/GaAs/phono3py/2x2x2/POSCAR"
    # poscar = "/Volumes/GoogleDrive/My Drive/multiphonon/rubrene_POSCAR"
    fc = "/Volumes/GoogleDrive/My Drive/Cori_backup/GaAs/phono3py/2x2x2/fc2.hdf5"
    # fc = "/Volumes/GoogleDrive/My Drive/multiphonon/rubrene_FORCE_SETS"
    fc3 = "/Volumes/GoogleDrive/My Drive/Cori_backup/GaAs/phono3py/2x2x2/FORCES_FC3"
    disp = "/Volumes/GoogleDrive/My Drive/Cori_backup/GaAs/phono3py/2x2x2/disp_fc3.yaml"
    #fc3 = None
    #disp = None
    born = "/Volumes/GoogleDrive/My Drive/Cori_backup/GaAs/phono3py/2x2x2/BORN"

    poscar = "/Users/tfharrelson/PycharmProjects/compute_Sqw/data/CsI/POSCAR"
    fc = "/Users/tfharrelson/PycharmProjects/compute_Sqw/data/CsI/FORCES_FC2"
    fc3 = "/Users/tfharrelson/PycharmProjects/compute_Sqw/data/CsI/FORCES_FC3"
    disp = "/Users/tfharrelson/PycharmProjects/compute_Sqw/data/CsI/disp_fc3.yaml"
    # fc3 = None
    # disp = None
    born = "/Users/tfharrelson/PycharmProjects/compute_Sqw/data/CsI/BORN"
    dynamic_structure_factor = DynamicStructureFactor(poscar,
                                                      fc,
                                                      mesh,
                                                      supercell,
                                                      # q_point_list=reduced_qpt[0, :].reshape(-1, 3),
                                                      q_point_list=qpoints,
                                                      # q_point_list=reduced_qpt,
                                                      max_e=max_e,
                                                      delta_e=delta_e,
                                                      num_overtones=num_overtones,
                                                      fc3_file=fc3,
                                                      fc3_disp=disp,
                                                      is_nac=True,
                                                      born_file=born,
                                                      scalar_mediator_flag=False,
                                                      dark_photon_flag=True)
    dynamic_structure_factor.get_coherent_sqw()
    dynamic_structure_factor.write_coherent_sqw('csi_darkphoton_sqw_m999_anh.hdf5')
    # s_qw = compute_Sqw(phonons, reduced_qpt, delta_e, max_e, num_overtones)
    # s_qw = compute_incoherent_Sqw(phonons, reduced_qpt, delta_e, max_e, num_overtones)
    # s_qw, decoherence_time = compute_decoherence_time(phonons, reduced_qpt, delta_e, max_e, num_overtones, anh_phonons.gammas, True)
