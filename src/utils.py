import numpy as np
from phonopy.interface.vasp import read_vasp
from phono3py.api_phono3py import Phono3py
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
                 disp_file='disp.yaml',
                 mesh=[5, 5, 5],
                 shift=[0., 0., 0.],
                 supercell=[2, 2, 2],
                 nac='False',
                 temperature=0.):
        self.poscar = poscar
        self.fc3_file = fc3_file
        self.disp_file = disp_file
        self.mesh = mesh
        self.shift = shift
        self.supercell = supercell
        self.nac = nac
        self.temperature = temperature
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
        #key = tuple(self._shift_q_to_1stBZ(qpoint))
        #key += (band_index,)
        key = self.set_key(self._shift_q_to_1stBZ(qpoint), band_index)
        return self.property_dict[key]

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
        self.irr_BZ_gridpoints = None
        self.phonon_freqs = None
        self.temperature = temperature
        # self.phono3py.run_imag_self_energy(np.unique(self.mapping), temperatures=temperature)

    def set_irr_BZ_gridpoints(self):
        self.mapping, grid = get_ir_reciprocal_mesh(mesh=self.mesh, cell=self.cell)
        self.grid = {tuple(k / self.mesh): v for (v, k) in enumerate(grid)}
        irr_BZ_gridpoints = np.unique(self.mapping)
        self.irr_BZ_gridpoints = {k: v for v, k in enumerate(irr_BZ_gridpoints)}

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
                                 supercell_matrix=self.inputs.supercell,
                                 primitive_matrix='auto',
                                 mesh=self.inputs.mesh,
                                 log_level=1)
        self.disp_data = parse_disp_fc3_yaml(filename=self.inputs.disp_file)
        self.fc3_data = parse_FORCES_FC3(self.disp_data, filename=self.inputs.fc3_file)
        with suppress_stdout():
            self.phono3py.produce_fc3(self.fc3_data,
                                      displacement_dataset=self.disp_data,
                                      symmetrize_fc3r=True)
        ## initialize phonon-phonon interaction instance
        self.phono3py.init_phph_interaction()
        # initialize bands, eigvecs, ise, and qpoints
        self.bands = None
        self.eigvecs = None
        self.imag_self_energy = None
        self.qpoints = None

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
        # get Phono3py manager to help
        #self.mapping = None
        #self.grid = None
        #self.irr_BZ_gridpoints = None
        #self.temperature = temperature
        # self.phono3py.run_imag_self_energy(np.unique(self.mapping), temperatures=temperature)
        self.imag_self_energy = {}
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
        return self.manager.phono3py._frequency_points[grid_index][0], self.manager.phono3py._imag_self_energy[grid_index][0][0, :, :]

    def get_imag_self_energies_at_q(self, qpoint):
        gridpoint = self._brillouinzone.grid[tuple(qpoint)]
        _, ises = self._get_imag_self_energies_from_gp(gridpoint)
        return ises

    def set_property_dict(self):
        for i, q in enumerate(self._brillouinzone.qpoints):
            # i is the gridpoint index
            # q is the qpoint vector
            self.freqs, band_of_ise = self._get_imag_self_energies_from_gp(i)
            # loop over bands which are the last index of 'bands_of_ise' matrix
            for band_index in range(band_of_ise.shape[-1]):
                #key = tuple(q)
                #key += (band_index,)
                key = self.set_key(q, band_index)
                self.property_dict[key] = band_of_ise[:, band_index]
"""
    def get_imag_self_energy(self, qpoint, band_index):
        if len(self.imag_self_energy) == 0:
            self.set_ise_dict()
        key = tuple(qpoint)
        key += (band_index,)
        return self.imag_self_energy[key]
"""


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
"""
    def get_eigenvalue(self, qpoint, band_index):
        if len(self.eigenvalues) == 0:
            self.set_eig_dict()
        key = tuple(qpoint)
        key += (band_index,)
        return self.eigenvalues[key]
"""


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
                #key = tuple(q)
                #key += (band_index,)
                key = self.set_key(q, band_index)
                self.property_dict[key] = eigvec
"""
    def get_eigenvector(self, qpoint, band_index):
        if len(self.eigenvectors) == 0:
            self.set_eigvec_dict()
        key = tuple(qpoint)
        key += (band_index,)
        return self.eigenvectors[key]
"""
