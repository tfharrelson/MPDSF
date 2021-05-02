import numpy as np
from scipy.interpolate import interp1d
from phonopy.interface.vasp import read_vasp
from phonopy.file_IO import parse_BORN
from phonopy.units import Bohr, Hartree
from phonopy.phonon.group_velocity import GroupVelocity
from phono3py.api_phono3py import Phono3py
from phono3py.api_isotope import Phono3pyIsotope
from phono3py.file_IO import (parse_disp_fc3_yaml,
                              parse_FORCES_FC3)
from spglib import get_ir_reciprocal_mesh
from contextlib import contextmanager
import sys, os


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


class Phono3pyInputs:
    def __init__(self,
                 poscar='POSCAR',
                 fc3_file='FORCES_FC3',
                 fc2_file='FORCE_SETS',
                 disp_file='disp.yaml',
                 mesh=[5, 5, 5],
                 shift=[0., 0., 0.],
                 supercell=[2, 2, 2],
                 nac=False,
                 born_file=None,
                 temperature=0.,
                 isotope_flag=False):
        self.poscar = poscar
        self.fc2_file = fc2_file
        self.fc3_file = fc3_file
        self.disp_file = disp_file
        self.mesh = mesh
        self.shift = shift
        self.supercell = supercell
        self.nac = nac
        self.born_file = born_file
        self.temperature = temperature
        self.isotope_flag = isotope_flag
        pass


class WeightedQuantity:
    """
    Abstract class that covers all phonon-based numbers that have k-point and branch indices. Examples include
    phonon band frequencies, eigenvectors, imaginary self energies

    AS FAR AS I KNOW: Phono3py naturally makes phonon objects on the entire BZ, not the irr BZ, which is kind of
    annoying. But I will leave this class here in case this changes, and Phono3py naturally outputs the phonon data on
    the irr BZ points.
    """
    def __init__(self, property=None, weight=1):
        self.property = property
        self.weight = weight


class MPGrid:
    def __init__(self, mesh, shift=None):
        self.mesh = mesh
        if shift is not None:
            self.shift = shift
        else:
            self.shift = [0., 0., 0.]
        self.qpoints = None
        self.padded_qpoints = None

        #set qpoints from mesh and shift
        if self.mesh is not None:
            self.set_qpoints()

    def set_qpoints(self):
        self.qpoints = np.zeros([np.prod(self.mesh), 3])
        count = 0
        curr_qx = self.shift[0]
        curr_qy = self.shift[1]
        curr_qz = self.shift[2]
        spacing = 1.0 / np.array(self.mesh)
        for z in range(self.mesh[2]):
            for y in range(self.mesh[1]):
                for x in range(self.mesh[0]):
                    self.qpoints[count, :] = [curr_qx, curr_qy, curr_qz]
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

    def set_padded_qpoints(self):
        self.padded_qpoints = np.zeros([np.prod(np.array(self.mesh) + 2), 3])
        count = 0
        curr_qx = self.shift[0]
        curr_qy = self.shift[1]
        curr_qz = self.shift[2]
        spacing = 1.0 / np.array(self.mesh)
        for z in range(self.mesh[2]):
            for y in range(self.mesh[1]):
                for x in range(self.mesh[0]):
                    self.padded_qpoints[count, :] = [curr_qx, curr_qy, curr_qz]
                    count += 1
                    curr_qx += spacing[0]
                    if curr_qx > 0.5 + spacing[0]:
                        curr_qx -= (1.0 + 2 * spacing[0])
                curr_qy += spacing[1]
                if curr_qy > 0.5 + spacing[1]:
                    curr_qy -= (1.0 + 2 * spacing[1])
            curr_qz += spacing[2]
            if curr_qz > 0.5 + spacing[2]:
                curr_qz -= (1.0 + 2 * spacing[2])

    def get_padded_qpoints(self):
        if self.padded_qpoints is None:
            self.set_padded_qpoints()
        return self.padded_qpoints


class PhaseSpace(MPGrid):
    def __init__(self, freqs=None, **kwargs):
        super().__init__(**kwargs)
        self.freqs = freqs
        self.phase_space = None
        self.padded_phase_space = None
        self.set_phase_space()

    def set_phase_space(self):
        # set phase_space from freqs and qpoints from super()
        if self.freqs is not None:
            self.phase_space = np.array([list(q) + [freq] for q in self.qpoints for freq in self.freqs])
        else:
            # if freqs is not given, then PhaseSpace object acts exactly like an MPGrid object
            self.phase_space = self.qpoints

    def set_padded_phase_space(self):
        # set phase_space from freqs and qpoints from super()
        if self.freqs is not None:
            if self.padded_qpoints is None:
                self.set_padded_qpoints()
            self.padded_phase_space = np.array([list(q) + [freq] for q in self.padded_qpoints for freq in self.freqs])
        else:
            # if freqs is not given, then PhaseSpace object acts exactly like an MPGrid object
            self.padded_phase_space = self.padded_qpoints

    def get_phase_space(self):
        if self.phase_space is None:
            self.set_phase_space()
        return self.phase_space

    def get_padded_phase_space(self):
        if self.padded_phase_space is None:
            self.set_padded_phase_space()
        return self.padded_phase_space


class BrillouinZoneProperty:
    """
    Abstract class that maps a specific property (band, eigenvector, imag self energy) to BZ coordinates and branch
    index. Default is to cast property over specified MP grid.
    """
    def __init__(self, inputs: Phono3pyInputs):
        self.mesh = inputs.mesh
        self.shift = inputs.shift
        self._brillouinzone = BrillouinZone(poscar=inputs.poscar, mesh=self.mesh, shift=self.shift)
        self.property = None
        self.manager = Phono3pyManager(inputs)
        self.input = inputs
        self.property_dict = {}
        # freqs is important to track for properties that have frequency dependence
        # if NoneType, then the property has no frequency dependence
        self.freqs = None
        self._interpolator = None

    def _set_interpolator(self):
        from src.Interpolation import Interpolator
        self._interpolator = Interpolator(phonon_property=self)

    def assign_value(self, key, value):
        self.property[key] = value

    def _shift_q_to_1stBZ(self, qpoint):
        shifted_qpoint = []
        for q in qpoint:
            if q <= -0.5:
                q += 1.
            elif q > 0.5:
                q -= 1.
            shifted_qpoint.append(q)
        return np.array(shifted_qpoint)

    def set_key(self, qpoint, band_index):
        key = []
        for q, s, m in zip(qpoint, self.shift, self.mesh):
            key.append(np.round((q - s) * m).astype(int))
        key.append(band_index)
        return tuple(key)

    def get_property_value(self, qpoint, band_index):
        if len(self.property_dict) == 0:
            self.set_property_dict()
        key = self.set_key(self._shift_q_to_1stBZ(qpoint), band_index)
        try:
            # try to find property at grid q-point and mode index
            return self.property_dict[key]
        except:
            # use interpolator
            if self._interpolator is None:
                self._set_interpolator()
            if self.freqs is None:
                return self._interpolator.interpolate(band_index, *qpoint)
            else:
                # prepare frequency dep inputs
                args = np.array([list(qpoint) + [f] for f in self.freqs]).T
                # pass the args into the interpolator
                return self._interpolator(band_index, *args)


    def set_property_dict(self):
        pass

class BrillouinZone(MPGrid):
    """
    Abstract class for organizing all annoying gridpoints and mappings within Phono3py
    """
    def __init__(self, poscar='POSCAR', temperature=0., **kwargs):
        super().__init__(**kwargs)
        self.cell = read_vasp(poscar)
        self.mapping = None
        self.grid = None
        self.inverse_grid = None
        self.irr_BZ_gridpoints = None
        self.phonon_freqs = None
        self.temperature = temperature
        self.irr_BZ_qpoints = None
        self.weights = None
        self._init_BZ()

    def _init_BZ(self):
        self.mapping, grid = get_ir_reciprocal_mesh(mesh=self.mesh, cell=self.cell)
        self.grid = {tuple(k / self.mesh): v for (v, k) in enumerate(grid)}
        self.inverse_grid = {v: tuple(k / self.mesh) for (v, k) in enumerate(grid)}
        irr_BZ_gridpoints = np.unique(self.mapping)
        self.irr_BZ_gridpoints = {k: v for v, k in enumerate(irr_BZ_gridpoints)}

        # Create a dict of irreducible q-points; key is a q-point in full grid, and value is the irreducible q-point
        self.irr_BZ_qpoints = {k: self.get_qpoint(self.mapping[gp]) for k, gp in self.grid.items()}
        # Create dictionary of weights for each irred q-point
        self.weights = {self.get_qpoint(irred_gp): list(self.mapping).count(irred_gp)
                        for irred_gp in irr_BZ_gridpoints}

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

    def get_qpoint(self, gridpoint):
        return self.inverse_grid[gridpoint]

class Phono3pyManager:
    def __init__(self,
                 inputs: Phono3pyInputs):
        """
        Phono3pyManager is a class that manages all relevant objects produced by Phono3py like phonon bands,
        eigenvectors, and imag self energies.
        :param poscar: Poscar file name from vasp calculation
        :param fc3_file: Force constants file used in Phono3py calculation, only FORCES_FC3 format usable now
        :param disp_file: The disp.yaml file from phono3py -d run
        :param mesh: The Monkhorst-Pack mesh grid (type=list)
        :param supercell: Supercell used for phonon calculation (type=list)
        """
        self.inputs = inputs
        self.cell = read_vasp(self.inputs.poscar)
        #self.mesh = mesh
        self.phono3py = Phono3py(self.cell,
                                 supercell_matrix=np.diag(self.inputs.supercell),
                                 primitive_matrix='auto',
                                 mesh=self.inputs.mesh,
                                 log_level=1)
        self.disp_data = parse_disp_fc3_yaml(filename=self.inputs.disp_file)
        self.fc3_data = parse_FORCES_FC3(self.disp_data, filename=self.inputs.fc3_file)
        with suppress_stdout():
            self.phono3py.produce_fc3(self.fc3_data,
                                      displacement_dataset=self.disp_data,
                                      symmetrize_fc3r=True)
        self.nac_params = None
        if inputs.nac is True:
            primitive = self.phono3py.get_phonon_primitive()
            self.nac_params = parse_BORN(primitive, filename=inputs.born_file)
            self.nac_params['factor'] = Hartree * Bohr
            self.phono3py.nac_params = self.nac_params
        ## initialize phonon-phonon interaction instance
        self.phono3py.init_phph_interaction()
        # initialize bands, eigvecs, ise, and qpoints
        self.bands = None
        self.eigvecs = None
        self.imag_self_energy = None
        self.qpoints = None
        if self.inputs.isotope_flag:
            primitive = self.phono3py.get_phonon_primitive()
            self.isotopes = Phono3pyIsotope(self.inputs.mesh, primitive, cutoff_frequency=1.e-4)
        else:
            self.isotopes = None

    def set_phonons(self):
        self.phono3py.run_phonon_solver()
        self.bands, self.eigvecs, self.qpoints = self.phono3py.get_phonon_data()
        self.fix_phonon_data()

    def fix_phonon_data(self):
        # check if there are more than the expected number of qpoints
        expected_num_qpoints = np.round(np.prod(self.inputs.mesh)).astype(int)
        if expected_num_qpoints < len(self.qpoints):
            # if true, then Phono3py is including redundant q-points outside of 1st Brillouin zone at the end of each
            # dataset
            self.bands = self.bands[:expected_num_qpoints, :]
            self.eigvecs = self.eigvecs[:expected_num_qpoints, :, :]
            self.qpoints = self.qpoints[:expected_num_qpoints, :]
        # want to ensure all q-points are inside the 1st Brillouin zone
        self.center_qpoints()
    def center_qpoints(self):
        # Move any q-points outside the first Brillouin zone back into the first zone
        temp_qpoints = []
        for q in self.qpoints:
            q = np.array(q).astype(np.float64)
            q /= self.inputs.mesh
            for ind, qx in enumerate(q):
                if qx / self.inputs.mesh[ind] > 0.5:
                    qx -= 1.
                elif qx / self.inputs.mesh[ind] < -0.5:
                    qx += 1.
                q[ind] = qx
            temp_qpoints.append(q)
        self.qpoints = temp_qpoints

class ImaginarySelfEnergy(BrillouinZoneProperty):
    def __init__(self, inputs: Phono3pyInputs):
        """
        ImaginarySelfEnergy is effectively a wrapper of the equivalent Phono3py object. It uses Phono3py to compute the
        imaginary self energy function, and some function calls are implemented to expedite the retrieval of relevant
        information.
        :param poscar: Poscar file name from vasp calculation
        :param fc3_file: Force constants file used in Phono3py calculation, only FORCES_FC3 format usable now
        :param disp_file: The disp.yaml file from phono3py -d run
        :param mesh: The Monkhorst-Pack mesh grid (type=list)
        :param supercell: Supercell used for phonon calculation (type=list)
        :param temperature: Temperature for phonon calculations (type=float)
        """
        super().__init__(inputs)
        self.imag_self_energy = {}
        self.freqs = None
        self.set_self_energies()
        self.set_property_dict()

    def set_self_energies(self):
        #########################################################################################################
        # imaginary self energy objects are stored in phono3py object in a weird way                            #
        # the object is a list of a list of ndarrays                                                            #
        # first index = grid-points, which are "addresses" of the irreducible q-points in the Brillouin zone    #
        #       At the surface these are meaningless, but the grid points are given by:                         #
        #       np.unique(mapping)                                                                              #
        #       The actual q-points are stored in grid with a 1-to-1 correspondence to mapping                  #
        # second index = sigma values, there is only one sigma=None in default tetrahedron method               #
        # third index is the nparray, which is arranged as:                                                     #
        #       (temperatures, frequency points, band index)                                                    #
        #########################################################################################################
        self.manager.phono3py.init_phph_interaction()
        if self._brillouinzone.mapping is None:
            self._brillouinzone.set_irr_BZ_gridpoints()
        if type(self.input.temperature) is not list:
            temp = [self.input.temperature]
        else:
            temp = self.input.temperature
        with suppress_stdout():
            self.manager.phono3py.run_imag_self_energy(np.unique(self._brillouinzone.mapping), temperatures=temp)

    def _get_imag_self_energies_from_gp(self, gridpoint):
        #########################################################################################################
        # Note that this returns the imaginary self energy function                                             #
        # Does not return the self energy evaluated at the phonon frequency                                     #
        # As a result, a tuple is returned with the frequency points, and self energy                           #
        #########################################################################################################
        irr_gp = self._brillouinzone.mapping[gridpoint]
        grid_index = self._brillouinzone.irr_BZ_gridpoints[irr_gp]
        return self.manager.phono3py._frequency_points, self.manager.phono3py._gammas[grid_index][0][0, :, :]

    def get_imag_self_energies_at_q(self, qpoint):
        gridpoint = self._brillouinzone.grid[tuple(qpoint)]
        _, ises = self._get_imag_self_energies_from_gp(gridpoint)
        return ises

    def set_property_dict(self):
        for i, q in enumerate(self._brillouinzone.qpoints):
            # i is the gridpoint index
            # q is the qpoint vector
            self.freqs, band_of_ise = self._get_imag_self_energies_from_gp(i)
            # loop over bands which are the 2nd to last index of 'bands_of_ise' matrix
            for band_index in range(band_of_ise.shape[-2]):
                #key = tuple(q)
                #key += (band_index,)
                key = self.set_key(q, band_index)
                self.property_dict[key] = band_of_ise[band_index, :]


class PhononEigenvalues(BrillouinZoneProperty):
    def __init__(self, inputs: Phono3pyInputs):
        super().__init__(inputs)
        self.eigenvalues = {}
        self.set_property_dict()

    def set_property_dict(self):
        self.manager.set_phonons()
        for i, q in enumerate(self._brillouinzone.qpoints):
            for band_index, eig in enumerate(self.manager.bands[i, :]):
                #key = tuple(q)
                #key += (band_index,)
                key = self.set_key(q, band_index)
                self.property_dict[key] = eig


class PhononEigenvectors(BrillouinZoneProperty):
    def __init__(self, inputs: Phono3pyInputs):
        super().__init__(inputs)
        self.eigenvectors = {}
        self.set_property_dict()

    def set_property_dict(self):
        # IMPORTANT: I believe the way Phono3py stores it's eigenvectors is in a matrix of shape:
        # (num qpts, num bands, 3 * num_atoms)
        # This is hard to test because the shape of the matrix at each q-point is square; (num bands) = (3 * num atoms)
        # However, in the Phono3py source code, the eigenvectors are determined via a call to linalg.eigh acting on the
        # dynamical matrix. This call returns the eigenvectors as column vectors in a square matrix such that the second
        # index is eigenvector index, and the first index corresponds to the eigenvector components

        self.manager.set_phonons()
        for i, q in enumerate(self._brillouinzone.qpoints):
            for band_index, eigvec in enumerate(self.manager.eigvecs[i].T):
                # impossible to know which index is actually the eigenvector b/c it is a square matrix
                key = self.set_key(q, band_index)
                self.property_dict[key] = eigvec


class GroupVelocities(BrillouinZoneProperty):
    def __init__(self, inputs: Phono3pyInputs):
        super().__init__(inputs)
        self.group_velocities = {}
        self.set_property_dict()

    def set_property_dict(self):
        self.manager.phono3py.init_phph_interaction()
        group_vels = GroupVelocity(self.manager.phono3py.dynamical_matrix)

        group_vels.run(self._brillouinzone.qpoints)

        for q, gvs_at_q in zip(self._brillouinzone.qpoints, group_vels.group_velocities):
            for branch_index, gv in enumerate(gvs_at_q):
                key = self.set_key(qpoint=q, band_index=branch_index)
                self.property_dict[key] = gv



class IsotopicImagSelfEnergy(BrillouinZoneProperty):
    def __init__(self, inputs: Phono3pyInputs):
        super().__init__(inputs)
        self.ise = None
        self.units = 'THz = 1 / (4pi * tau)'
        self.imag_self_energy = {}
        self.set_self_energies()
        self.set_property_dict()

    def set_self_energies(self):
        '''
        Set isotopic self energies using PHono3pyIsotope class. The self energies are stored in
        self.manager.isotope.gamma with units of THz
        :return:
        '''
        # init dynamical matrix
        self.manager.isotopes.init_dynamical_matrix(fc2=self.manager.phono3py.fc2,
                                                    supercell=self.manager.phono3py.supercell,
                                                    primitive=self.manager.phono3py.primitive,
                                                    nac_params=self.manager.nac_params)
        if self._brillouinzone.mapping is None:
            self._brillouinzone.set_irr_BZ_gridpoints()
        gridpoints = [self._brillouinzone.get_gridpoint(q) for q in self._brillouinzone.qpoints]
        self.manager.isotopes.run(gridpoints)

    def set_property_dict(self):
        for i, q in enumerate(self._brillouinzone.qpoints):
            # i is the gridpoint index
            # q is the qpoint vector
            for band_index in range(np.array(self.manager.isotopes.gamma).shape[-1]):
                key = self.set_key(q, band_index)
                gridpoint = self._brillouinzone.get_gridpoint(q)
                # gamma matrix has three indices: (# sigmas, # gridpoints, # bands)
                # There is always only one sigma for these objects
                self.property_dict[key] = self.manager.isotopes.gamma[0][gridpoint][band_index]


class Gamma(BrillouinZoneProperty):
    '''
    This class is similar to ImaginarySelfEnergy, but evaluates the ISE at the phonon frequency only
    '''
    def __init__(self, inputs: Phono3pyInputs, f_min=1e-3):
        super().__init__(inputs)
        self.ise = ImaginarySelfEnergy(inputs)
        self.eigs = PhononEigenvalues(inputs)
        self.f_min = f_min
        self.set_property_dict()

    def set_property_dict(self):
        for key, ise_at_key in self.ise.property_dict.items():
            eig_at_key = self.eigs.property_dict[key]
            if eig_at_key < self.f_min:
                self.property_dict[key] = 0.
                continue
            interp = interp1d(self.ise.freqs, ise_at_key)
            self.property_dict[key] = interp(eig_at_key)