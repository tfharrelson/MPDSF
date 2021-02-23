import numpy as np
from phonopy import load
from src.utils import BrillouinZoneProperty, PhononEigenvalues, Phono3pyInputs, ImaginarySelfEnergy, \
    IsotopicImagSelfEnergy, GroupVelocities
import scipy.constants as const

'''
The goal of this script is to numerically compute the temperature variation in a readout device using a simple model.
The model effectively treats the readout device as a material with a source (from the target material) and a drain
(from the thermal bath). The drain is a simple heat transfer problem: W = G * (T_r - T_b), where W is the power, G is 
the heat conductance, T_r is the temperature of the readout device, and T_b is the bath temperature. The source is more
complicated and requires numerical integration of a heat flux according to an equation:
    
    Phi = 1/2 integral( hbar * w * v_g(w) * D(w) * alpha * (g(w, t) - f(w, T)) ) wrt omega
    
    w is the frequency of the dynamics, v_g is the mean group velocity of the phonons at omega, D is the density 
    of states, alpha is the transmission coefficient averaged over all angles of incidence, g is the time-dependent
    non-equilibrium distribution function, and f is the equilibrium distribution function.
    
    We assume alpha to be ~0.5, but of course this is material dependent, and can vary between 0 - 1. All objects in the
    integral are easy to compute from first principles, with the exception of g(omega, t). We assume this takes the
    form:
        dg/dt = -Gamma_(anh,o) * g(w_o, t) * delta_(w,w_o) - Gamma_iso(w) * g(w, t) + Gamma_(anh,o) * P_o(w) * g(w, t)
    where the Gamma's are the inverse relaxation rates for anharmonic and isotopic scattering, and P_o(w) is the
    probability that the initial state converts to a new state with frequency w. P_o(w) is calculated in the 
    'plot_decay_channels' code. 
    
    This script computes the binned group velocity at each omega, the density of states, the binned Gamma_iso, and 
    P_o(w) using plot_decay_channels. Then each of these objects are used to create a time series of the heat flux
    into the readout device. Then, we use a Green's function approach to calculate the time series for the temperature
    in the readout device.
'''


# For now, write classes here before moving them to src, leaving just the script
class ReadoutDevice:
    '''
    A class that contains the characteristics of a readout device. All defaults are taken from DOI: XXXXX
    :param heat_capacity - heat capacity of the readout device
    :param bath_conductance - thermal conductance for heat transfer between device and thermal bath
    :param target_area - interfacial area of interconnect between target (absorber) and readout device
    :param readout_volume - volume of readout device
    '''

    def __init__(self,
                 heat_capacity=None,
                 bath_conductance=None,
                 target_area=None,
                 readout_volume=None):
        # Set device specific characteristics
        self.heat_capacity = heat_capacity
        self.bath_conductance = bath_conductance
        self.target_area = target_area
        self.readout_volume = readout_volume


class DensityOfStates:
    def __init__(self,
                 inputs: Phono3pyInputs):
        self.inputs = inputs

        # Load into phonopy; hard-code the 'auto' primitive matrix option
        self.phonopy = load(supercell_matrix=self.inputs.supercell,
                            primitive_matrix='auto',
                            unitcell_filename=self.inputs.poscar,
                            force_constants_filename=self.inputs.fc2_file,
                            is_nac=self.inputs.nac,
                            born_filename=self.inputs.born_file)

    def get_total_DOS(self, mesh=None):
        if mesh is None:
            self.phonopy.run_mesh(self.inputs.mesh)
        else:
            self.phonopy.run_mesh(mesh)
        self.phonopy.run_total_dos()

        return self.phonopy.get_total_DOS()


class PropertyBinner:
    '''
    This class bins phonon-based properties. It can be changed for other BZ properties, but all the testing will be done
    for phonon-based properties. The biggest foreseen difference is that the energies associated with each property is
    positive semi-definite for phonons, and fully real for electron properties.
    '''

    def __init__(self,
                 property: BrillouinZoneProperty,
                 energies: PhononEigenvalues,
                 sigma=None):
        '''
        :param property: An BrillouinZoneProperty object of property values for each phonon branch and k-point index in the BZ.
        :param energies: An PhononEigenvalues of the phonon energy eigenvalues for each phonon branch and k-point index in the BZ.
        '''
        self.property = property
        self.energies = energies
        self.binned_property = None
        self.avg_binned_property = None
        self.bins = PropertyBins(energies)
        self.sigma = sigma

    def bin_property(self, bin_width=None):
        # Set the energy bins, allow user to set bin width if default behavior is not desired
        self.bins.set_energy_bins(bin_width=bin_width)

        # Initialize bins as empty lists
        self.initialize_bins()

        # Loop through all property values and place them in their respective bins
        for prop_key, prop_val in self.property.property_dict.items():
            energy = self.energies.property_dict[prop_key]
            bin_number = np.floor(energy / self.bins.bin_width).astype(int)
            self.binned_property[bin_number].append(prop_val)

    def average_property_bins(self, norm=True):
        '''
        Loop over binned properties and average over each bin
        :param norm: Boolean that describes whether the norm of the property is averaged.
        :return: set avg_binned_property in object field
        '''
        # Initialize for minor speed reasons
        self.avg_binned_property = np.zeros(len(self.bins.bin_energies))

        # Loop over binned_property and average appropriately
        for i, bp_vec in enumerate(self.binned_property):
            if len(bp_vec) == 0:
                self.avg_binned_property[i] = 0
            elif norm:
                if np.array(bp_vec).ndim > 1:
                    avg_binned_property = np.mean(np.linalg.norm(bp_vec, axis=1))
                else:
                    avg_binned_property = np.mean(bp_vec)

                if self.sigma is None:
                    self.avg_binned_property[i] = avg_binned_property
                else:
                    self.avg_binned_property += avg_binned_property * self.create_gaussian(self.bins.bin_energies[i])
            else:
                # code not set up for averages that don't find the norm of a vector quantity first.
                # I don't currently know what would be most useful, so here's a snarky error message.
                print("CONGRATS YOU BROKE IT, I HOPE YOU'RE HAPPY. ENJOY YOUR CRYPTIC ERROR")

    def initialize_bins(self):
        '''
        Initialize the binned_property object to an list of empty lists with a length equal to the number of bins
        :return: Nothing, set the binned_property field in the object.
        '''
        self.binned_property = [[] for _ in self.bins.bin_energies]

    def gaussian(self, x):
        if self.sigma is None:
            return None
        return 1 / np.sqrt(2 * np.pi * self.sigma ** 2) * np.exp(-1.0 * x ** 2 / (2 * self.sigma ** 2))

    def create_gaussian(self, mu):
        frequencies = self.bins.bin_energies
        return self.gaussian(frequencies - mu)


class PropertyBins:
    def __init__(self,
                 energies: PhononEigenvalues,
                 bin_width=None):
        self.energies = energies
        self.bin_energies = None
        self.bin_width = bin_width
        self.num_bins = None

        self.set_energy_bins()

    def set_energy_bins(self, bin_width=None):
        '''
        Set the number and width of the energy bins to be used in the binning procedure.
        :param bin_width: Sets the energy width of each bin. Default is None, for which the code automatically sets the width.
        :return: Nothing; the relevant values are stored as fields in the PropertyBinner object
        '''
        # Reshape the energies into a 1-D list and find its max
        max_e = max(self.energies.property_dict.values())

        # Use the max energy and the bin_width to find the number of bins and set the bins
        if bin_width is not None:
            self.num_bins = np.ceil(max_e / bin_width).astype(int)
            self.bin_width = bin_width
        else:
            # If bin_width not set, then set it automatically using the following heuristic:
            # We want the number of data points to be 10-fold the number of bins

            # Find the number of energies in the dataset by taking the len of property_dict of energies object
            num_energies = len(self.energies.property_dict)

            # Use this to set the number of bins using the above rule
            self.num_bins = np.ceil(num_energies / 20.).astype(int)

            # Use num_bins and max_e to find the bin_width
            # Multiply max_e by 1.01 to ensure it falls in the bins
            self.bin_width = 1.01 * max_e / self.num_bins

        # Use the set bin_widths and num_bins to find the bins
        self.bin_energies = np.arange(self.num_bins) * self.bin_width + (self.bin_width / 2)

    def get_bin_index(self, energy):
        '''
        Get the bin index for an arbitrary energy
        :param energy:
        :return: bin_index as an int or None if outside of range
        '''
        bin_index = np.floor(energy / self.bin_width).astype(int)
        if bin_index < 0 or bin_index >= len(self.bin_energies):
            return None
        else:
            return bin_index


class NonEqDistribution:
    def __init__(self,
                 bins: PropertyBins = None,
                 energies: PhononEigenvalues = None):
        self.curr_g = None
        self.past_g = []
        if bins is not None:
            self.bins = bins
        elif energies is not None:
            self.bins = PropertyBins(energies)
        else:
            print("Bins set to None during initialization of Noneq dist function. I hope you know what you're doing")
            self.bins = None
        if self.bins is not None:
            self.curr_g = np.zeros(len(self.bins.bin_energies))

    def initialize_distribution(self, energy, weight):
        '''
        Set the weight of the non-eq dist function at a specific energy.
        :param energy: energy at which to change the distribution function
        :param weight: new weight of dist. function at that energy
        :return: None
        '''
        # Find the bin_index for the inputted energy
        bin_index = self.bins.get_bin_index(energy)
        # Change the curr_g dist function at that energy bin to the specified weight
        self.curr_g[bin_index] = weight

    def update_distribution(self, new_dist):
        self.past_g.append(self.curr_g)
        self.curr_g = new_dist


class NoneqDistributionIntegrator:
    '''
    Takes an initial distribution object, and a set of rules, and updates to a new distribution function at the next
    time step.
    '''

    def __init__(self,
                 distribution: NonEqDistribution,
                 method='euler',
                 time_step=None,
                 num_steps=100):
        self.kernel = None
        self.distribution = distribution
        self.method = method
        self.time_step = time_step
        self.num_steps = num_steps
        self.curr_time = 0

    def set_kernel(self, kernel):
        if self.kernel is not None:
            composite_kernel = CompositeKernel(self.bins)
            self.kernel = composite_kernel.combine_kernels(self.kernel, kernel)
        else:
            self.kernel = kernel

    def integrate(self):
        if self.method is 'euler':
            self.integrate_euler()
        elif self.method is 'back_euler':
            self.integrate_back_euler()

    def integrate_back_euler(self):
        for _ in range(self.num_steps):
            self.distribution.update_distribution(self.integrate_back_euler_step())

    def integrate_back_euler_step(self):
        matrix = np.eye(self.distribution.bins.num_bins) - self.kernel.kernel_matrix * self.time_step
        b = self.distribution.curr_g
        return np.linalg.solve(matrix, b)

    def integrate_euler(self):
        for _ in range(self.num_steps):
            self.distribution.update_distribution(self.integrate_euler_step())

    def integrate_euler_step(self):
        # This method does not include a source term
        new_g = self.distribution.curr_g + self.time_step * np.dot(self.kernel.kernel_matrix, self.distribution.curr_g)
        return new_g


class IntegrationKernel:
    def __init__(self, bins: PropertyBins):
        self.kernel_matrix = None
        self.bins = bins
        self._init_kernel()

    def _init_kernel(self):
        self.kernel_matrix = np.zeros([self.bins.num_bins, self.bins.num_bins])

    def set_kernel(self):
        pass


class CompositeKernel(IntegrationKernel):
    def __init__(self, bins: PropertyBins):
        super().__init__(bins)

    def combine_kernels(self,
                        kernel_1: IntegrationKernel,
                        kernel_2: IntegrationKernel):
        self.kernel_matrix = kernel_1.kernel_matrix + kernel_2.kernel_matrix


class DiagonalKernel(IntegrationKernel):
    def __init__(self,
                 bins: PropertyBins,
                 diag_elements):
        super().__init__(bins=bins)
        self.diag_elements = diag_elements
        self.set_kernel()

    def set_kernel(self):
        self.kernel_matrix = np.diag(self.diag_elements)


class DecayKernel(IntegrationKernel):
    def __init__(self,
                 bins: PropertyBins,
                 decay_energy,
                 decay_weight):
        super().__init__(bins=bins)
        self.decay_energy = decay_energy
        self.decay_weight = decay_weight
        self.set_kernel()

    def set_kernel(self):
        bin_index = self.bins.get_bin_index(self.decay_energy)
        self.kernel_matrix[bin_index, bin_index] = self.decay_weight


class ChannelKernel(IntegrationKernel):
    def __init__(self,
                 bins: PropertyBins,
                 channel_energy,
                 channel_distribution):
        super().__init__(bins)
        self.channel_energy = channel_energy
        self.channel_distribution = channel_distribution
        self.set_kernel()

    def set_kernel(self):
        bin_index = self.bins.get_bin_index(self.channel_energy)
        self.kernel_matrix[:, bin_index] = self.channel_distribution


class DecayChannelKernel(CompositeKernel):
    def __init__(self,
                 bins: PropertyBins,
                 decay_energy,
                 channel_vec,
                 decay_weight=None):
        super().__init__(bins)
        if decay_weight is None:
            decay_weight = -1. * np.sum(channel_vec)
        self.decay_kernel = DecayKernel(bins,
                                        decay_energy=decay_energy,
                                        decay_weight=decay_weight)
        self.channel_kernel = ChannelKernel(bins,
                                            channel_energy=decay_energy,
                                            channel_distribution=channel_vec)
        self.set_kernel()

    def set_kernel(self):
        self.combine_kernels(self.decay_kernel, self.channel_kernel)


class DecayDistribution(BrillouinZoneProperty):
    def __init__(self,
                 inputs: Phono3pyInputs,
                 bins: PropertyBins,
                 min_freq=0.01,
                 temperature=1.,
                 sigma=None):
        '''
        Class that calculates the decay distribution of a phonon. Usage is to (1) instantiate the object, then (2)
        call get_decay_distribution_at_q(qpoint, branch) passing in the desired qpoint and branch index
        :param inputs: Phono3pyInputs object containing the location of all files required to run Phono3py
        :param bins: PropertyBins object that contains details about the bins/grid that the distribution lives on.
        :param min_freq: The minimum phonon frequency allowed -> avoids counting negative freq modes.
        '''
        super().__init__(inputs=inputs)
        self._brillouinzone.set_irr_BZ_gridpoints()
        self.decay_distribution = None
        self.decay_rate = None
        self.pp = None
        self.triplets = None
        self.triplet_map = None
        self.weights = None
        self.frequencies = PhononEigenvalues(inputs)
        self.sigma = sigma
        self.bins = bins
        self.min_freq = min_freq
        self.temperature = temperature

    def get_decay_distribution_at_q(self, qpoint, branch_index):
        # Get gridpoint from qpoint

        gridpoint = self._brillouinzone.get_gridpoint(qpoint)

        # Set gridpoint in Phono3py interaction
        pp = self.manager.phono3py.phph_interaction
        pp.set_grid_point(gridpoint, stores_triplets_map=True)

        # Run the Phono3py interaction code
        pp.run()

        # Get results
        self.pp = pp.get_interaction_strength()
        self.triplets, self.weights, self.triplet_map, ir_map_at_q = pp.get_triplets_at_q()

        # Use the results to calculate the decay distribution
        self.create_decay_spectrum(qpoint, branch_index)

    def get_gridpoints(self):
        return np.unique(self.triplet_map)

    def compute_occupation_number(self, frequency):
        return (np.exp(const.hbar * frequency * 10 ** 12 / (const.Boltzmann * self.temperature)) - 1) ** -1

    def gaussian(self, x):
        return 1 / np.sqrt(2 * np.pi * self.sigma ** 2) * np.exp(-1.0 * x ** 2 / (2 * self.sigma ** 2))

    def create_gaussian(self, mu):
        frequencies = self.bins.bin_energies
        return self.gaussian(frequencies - mu)

    def create_decay_spectrum(self,
                              qpoint,
                              branch_index,
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
        spectrum = np.zeros(self.bins.num_bins)
        decay_rate = 0.
        pp_shape = self.pp.shape
        min_factor = 1e-4 / self.bins.num_bins
        for tr in range(pp_shape[0]):
            for b2 in range(pp_shape[2]):
                for b3 in range(pp_shape[3]):
                    # tr = triplet index; b2 = branch index 2; b3 = branch index 3
                    pp_element = self.pp[tr, branch_index, b2, b3]
                    triplet = self.triplets[tr, :]

                    gp1_index = self.triplet_map[triplet[0]]
                    gp2_index = self.triplet_map[triplet[1]]
                    gp3_index = self.triplet_map[triplet[2]]

                    qpoint2 = self._brillouinzone.get_qpoint(gp2_index)
                    qpoint3 = self._brillouinzone.get_qpoint(gp3_index)

                    # combined_index_1 = get_combined_index(q1_index, pp_shape[1] - 1, num_branches)
                    # combined_index_2 = get_combined_index(q2_index, b2, num_branches)
                    # combined_index_3 = get_combined_index(q3_index, b3, num_branches)
                    # print('pp_shape =', pp.shape)
                    # if self.frequencies[combined_index_2] < min_freq:
                    if self.frequencies.get_property_value(qpoint=qpoint2, band_index=b2) < min_freq:
                        continue
                    # if self.frequencies[combined_index_3] < min_freq:
                    if self.frequencies.get_property_value(qpoint=qpoint3, band_index=b3) < min_freq:
                        continue

                    frequency2 = self.frequencies.get_property_value(qpoint2, b2)
                    frequency3 = self.frequencies.get_property_value(qpoint3, b3)

                    frequency1 = self.frequencies.get_property_value(qpoint, branch_index)
                    # combined_index_1 = get_combined_index(q1_index, pp_shape[1] - 1 - i, num_branches)
                    # if check_frequencies(frequencies[combined_index_1], frequencies[combined_index_2],
                    #                     frequencies[combined_index_3], thresh):
                    int_factor = self.gaussian(frequency1 - frequency2 - frequency3)
                    gaussian2 = self.create_gaussian(frequency2)
                    gaussian3 = self.create_gaussian(frequency3)
                    if int_factor > min_factor:
                        spectrum += self.weights[tr] * pp_element * gaussian2 * \
                                    (self.compute_occupation_number(frequency2) +
                                     self.compute_occupation_number(frequency3) + 1) * int_factor
                        spectrum += self.weights[tr] * pp_element * gaussian3 * \
                                    (self.compute_occupation_number(frequency2) +
                                     self.compute_occupation_number(frequency3) + 1) * int_factor
                        decay_rate += self.weights[tr] * pp_element * \
                                      (self.compute_occupation_number(frequency2) +
                                       self.compute_occupation_number(frequency3) + 1) * int_factor

                    int_factor = self.gaussian(frequency1 + frequency2 - frequency3)
                    if int_factor > min_factor:
                        spectrum += self.weights[tr] * pp_element * gaussian2 * \
                                    (self.compute_occupation_number(frequency2) -
                                     self.compute_occupation_number(frequency3)) * int_factor
                        spectrum += self.weights[tr] * pp_element * gaussian3 * \
                                    (self.compute_occupation_number(frequency2) -
                                     self.compute_occupation_number(frequency3)) * int_factor
                        decay_rate += self.weights[tr] * pp_element * \
                                      (self.compute_occupation_number(frequency2) -
                                       self.compute_occupation_number(frequency3)) * int_factor

                    int_factor = self.gaussian(frequency1 - frequency2 + frequency3)
                    if int_factor > min_factor:
                        spectrum += self.weights[tr] * pp_element * gaussian2 * \
                                    (-self.compute_occupation_number(frequency2) +
                                     self.compute_occupation_number(frequency3)) * int_factor
                        spectrum += self.weights[tr] * pp_element * gaussian3 * \
                                    (-self.compute_occupation_number(frequency2) +
                                     self.compute_occupation_number(frequency3)) * int_factor
                        decay_rate += self.weights[tr] * pp_element * \
                                      (-self.compute_occupation_number(frequency2) +
                                       self.compute_occupation_number(frequency3)) * int_factor
        # Convert from eV**2 THz to THz
        conv_factor = 18 * np.pi / const.hbar ** 2 * const.e ** 2 * (10 ** -12) ** 2 / ((2 * np.pi) ** 2)
        self.decay_distribution = spectrum * conv_factor
        # Find the decay rate by integrating the decay distribution
        # IMPORTANT: need to divide the conversion factor by 2 to get appropriate decay rate b/c we are assuming that
        #            all decays go from 1 phonon into 2 phonons.
        # self.decay_rate = np.trapz(spectrum, self.bins.bin_energies) * conv_factor / 2
        self.decay_rate = decay_rate * conv_factor


# How to put it together???
if __name__ == '__main__':
    # Set up bins first
    # First get PhononEigenvalues
    import os

    print(os.getcwd())
    predir = '../data/GaAs/'
    poscar = predir + 'POSCAR'
    fc2 = predir + 'fc2.hdf5'
    fc3 = predir + 'FORCES_FC3'
    disp = predir + 'disp_fc3.yaml'
    born_file = predir + 'BORN'
    mesh = [7, 7, 7]
    supercell = [2, 2, 2]
    nac = True
    sigma=0.3

    gaas_inputs = Phono3pyInputs(poscar=poscar,
                                 fc2_file=fc2,
                                 fc3_file=fc3,
                                 disp_file=disp,
                                 mesh=mesh,
                                 supercell=supercell,
                                 nac=nac,
                                 born_file=born_file)
    eigs = PhononEigenvalues(gaas_inputs)

    # Get Imag Self Energies
    ise = ImaginarySelfEnergy(gaas_inputs)

    # Now set up PropertyBins
    bins = PropertyBins(eigs)

    # Also get DOS
    dos_obj = DensityOfStates(gaas_inputs)
    dos = dos_obj.get_total_DOS()
    # Interpolate dos onto bins
    from scipy.interpolate import interp1d

    interpolator = interp1d(dos[0], dos[1], kind='linear')
    unitcell_vol = np.linalg.det(eigs.manager.phono3py.primitive.cell)
    dos_on_bins = interpolator(bins.bin_energies) / unitcell_vol

    # Use ISE to build decay and channel kernels
    # Need the channel distribution, which is similar to what we get from the method in 'plot_decay_channels.py'
    # In order to get this, we need to get the pp object, the triplets, and the triplet map
    decay_channel_dist = DecayDistribution(gaas_inputs, bins=bins, sigma=sigma)
    decay_channel_dist.get_decay_distribution_at_q(qpoint=[0, 0, 0], branch_index=5)

    # Use bins to create non-eq distribution
    noneq_dist = NonEqDistribution(bins)

    # initialize noneq_dist
    # Create 1 phonon in the 6th phonon mode (last optical mode) at gamma
    q = [0, 0, 0]
    branch_index = 5
    noneq_energy = eigs.get_property_value(q, branch_index)
    noneq_dist.initialize_distribution(noneq_energy, 1.)

    # Bin isotopic gamma
    iso_inputs = Phono3pyInputs(poscar=poscar,
                                fc2_file=fc2,
                                fc3_file=fc3,
                                disp_file=disp,
                                mesh=mesh,
                                supercell=supercell,
                                nac=nac,
                                born_file=born_file,
                                isotope_flag=True)
    iso_gamma = IsotopicImagSelfEnergy(inputs=iso_inputs)
    iso_binner = PropertyBinner(iso_gamma, eigs, sigma=sigma)
    # Bin the isotopes
    iso_binner.bin_property()
    # average over binned values
    iso_binner.average_property_bins()

    # Bin Group velocities, repeat procedure above
    group_vels = GroupVelocities(inputs=gaas_inputs)
    gv_binner = PropertyBinner(group_vels, eigs, sigma=sigma)
    gv_binner.bin_property()
    gv_binner.average_property_bins()

    # Create interaction kernels
    isotope_kernel = DiagonalKernel(bins=bins,
                                    diag_elements=-0. * iso_binner.avg_binned_property)
    decay_channel_kernel = DecayChannelKernel(bins=bins,
                                              decay_energy=noneq_energy,
                                              channel_vec=decay_channel_dist.decay_distribution,
                                              decay_weight=-1 * decay_channel_dist.decay_rate)
    combined_kernel = CompositeKernel(bins)
    combined_kernel.combine_kernels(isotope_kernel, decay_channel_kernel)

    # Create flux kernel describing flux across interface
    # First define some specific device characteristics
    area_tes_abs = 7.5 * 0.200        # units of mm^2
    area_tes_abs *= 10**14          # unit conversion from mm^2 to Angstrom^2
    vol_target = 50.                # unit is cm^3
    vol_target *= 10**24

    flux_rate_dist = 1 / 4 * gv_binner.avg_binned_property * dos_on_bins * area_tes_abs / vol_target
    flux_kernel = DiagonalKernel(bins=bins, diag_elements=-1. * flux_rate_dist * bins.bin_width)

    combined_kernel.combine_kernels(combined_kernel, flux_kernel)

    # Create integrator
    time_step = 1000000.
    num_steps = 10000
    integrator = NoneqDistributionIntegrator(distribution=noneq_dist,
                                             method='back_euler',
                                             time_step=time_step,
                                             num_steps=num_steps)
    integrator.set_kernel(combined_kernel)
    integrator.integrate()

    # Create temperature model
    # Create power time series, converting bin_energies to 1/s, yielding units of J / ps
    power_time_series = [np.trapz(flux_rate_dist * const.h * bins.bin_energies * 10**12 * g) for g in noneq_dist.past_g]
    power_time_series = np.array(power_time_series)

    # Define more device constants
    tes_heat_cap = 22.7 * 10**-12           # Units in J / K
    tes_bath_conductance = 7.5 * 10**-21    # Units of J / (ps * K)

    from scipy.integrate import cumtrapz
    # Write out times
    times = np.linspace(time_step, num_steps * time_step, num_steps) - time_step
    delta_T = cumtrapz(power_time_series / tes_heat_cap * np.exp(-tes_bath_conductance / tes_heat_cap * times), times)
