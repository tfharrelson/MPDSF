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
    '''
    Class that compartmentalizes all the Phono3py input files and tags. They get complicated very quickly.
    '''
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
        '''
        Constructor that takes all Phono3py inputs and stores them in the class.

        Parameters
        -------
        poscar : str
            VASP POSCAR file.
        fc3_file : str
            Third order force sets file created by Phono3py. Default is 'FORCES_FC3'.
        fc2_file : str
            Second order force sets file created by either Phonopy or Phono3py. Default is 'FORCE_SETS'.
        disp_file : str
            Yaml file outputted by Phono3py that specifies the displacements made to calculate the force sets. Default is 'disp.yaml'
        mesh : List
            List of integers specifying a Gamma-centered Monkhorst-Pack grid for discretizing k-points. Default is [5,5,5].
        shift : List
            List of floats specifying the shift from a Gamma-centered grid.
        supercell : List
            List of integers specifying the size of the supercell created to calculate the force sets.
        nac : bool
            Boolean flag specifying if the non-analytic term correction is to be used.
        born_file : str
            BORN file created from `phonopy-vasp-born`.
        temperature : float
            Temperature of the material. Default 0 K.
        isotope_flag : bool
            Boolean flag specifying if isotopic scattering is to be calculated.
        '''
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
    '''
    Class that accounts for all details associated with the Monkhorst-Pack discretization of the Brilloin zone.
    '''
    def __init__(self, mesh, shift=None):
        '''
        Constructor from a mesh and shift list objects.

        Parameters
        -------
        mesh : List
            List of integers specifying a Gamma-centered Monkhorst-Pack grid
        shift : List
            List of floats specifying the shift from a Gamma-centered grid.
        '''
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
        '''
        Setter for q-points that depends on the mesh and shift parameters passed during contruction. Sets self.qpoints object.
        '''
        self.qpoints = np.zeros([np.prod(self.mesh), 3])
        count = 0
        curr_qx = self.shift[0]
        curr_qy = self.shift[1]
        curr_qz = self.shift[2]
        spacing = 1.0 / np.array(self.mesh)
        for z in range(self.mesh[2]):
            if z / self.mesh[2] + self.shift[2] > 0.5:
                adj_z = z - self.mesh[2]
            else:
                adj_z = z + self.shift[2]
            for y in range(self.mesh[1]):
                if y / self.mesh[1] + self.shift[1] > 0.5:
                    adj_y = y - self.mesh[1]
                else:
                    adj_y = y + self.shift[1]
                for x in range(self.mesh[0]):
                    if x / self.mesh[0] + self.shift[0] > 0.5:
                        adj_x = x - self.mesh[0]
                    else:
                        adj_x = x + self.shift[0]
                    self.qpoints[count, :] = np.array([adj_x, adj_y, adj_z]) / self.mesh + self.shift
                    #self.qpoints[count, :] = np.array([curr_qx, curr_qy, curr_qz])
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
        '''
        Setter for padded q-points. Padded by including redundant q-points related by periodicity of the Brillouin zone. For example, a 5x5x5 grid should only include q-points up to +2/5 and -2/5 in any direction, but a padded zone will have the 3/5 point which is the same as -2/5, and the -3/5 point (which is the 2/5 point).
        '''
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
        '''
        Getter class for padded q-points
        '''
        if self.padded_qpoints is None:
            self.set_padded_qpoints()
        return self.padded_qpoints


class PhaseSpace(MPGrid):
    '''
    Child class of MPGrid, which is extended to include frequency discretization, meaning that this class contains info on all q-points (momentum space) and frequencies (energy space), which is why it is called a phase space.
    '''
    def __init__(self, freqs=None, **kwargs):
        '''
        Constructor for PhaseSpace.

        Parameters
        -------
        freqs : List
            List of frequencies to be used in the phase space.
        kwargs : keyword args to be used in MPGrid construction.
        '''
        super().__init__(**kwargs)
        self.freqs = freqs
        self.phase_space = None
        self.padded_phase_space = None
        self.set_phase_space()

    def set_phase_space(self):
        '''
        Setter of the phase space object. phase_space object is Nx4 array for (qx, qy, qz, w) where N is the total number of phase-space points (equal to prod(mesh) * len(freqs)).
        '''
        # set phase_space from freqs and qpoints from super()
        if self.freqs is not None:
            self.phase_space = np.array([list(q) + [freq] for q in self.qpoints for freq in self.freqs])
        else:
            # if freqs is not given, then PhaseSpace object acts exactly like an MPGrid object
            self.phase_space = self.qpoints

    def set_padded_phase_space(self):
        '''
        Setter of padded phase space. Code is set up in a lazy way in which the phase-space objects are not calculated until necessary.
        '''
        # set phase_space from freqs and qpoints from super()
        if self.freqs is not None:
            if self.padded_qpoints is None:
                self.set_padded_qpoints()
            self.padded_phase_space = np.array([list(q) + [freq] for q in self.padded_qpoints for freq in self.freqs])
        else:
            # if freqs is not given, then PhaseSpace object acts exactly like an MPGrid object
            self.padded_phase_space = self.padded_qpoints

    def get_phase_space(self):
        '''
        Getter for phase space. If phase-space not set, it will be set by calling this.
        '''
        if self.phase_space is None:
            self.set_phase_space()
        return self.phase_space

    def get_padded_phase_space(self):
        '''
        Getter for padded phase space object.
        '''
        if self.padded_phase_space is None:
            self.set_padded_phase_space()
        return self.padded_phase_space


class BrillouinZoneProperty:
    """
    Abstract class that maps a specific property (band, eigenvector, imag self energy) to BZ coordinates and branch index. Default is to cast property over specified MP grid. Properties are stored as a dictionary, in which the keys are the q-point and band index combinations.
    """
    def __init__(self, inputs: Phono3pyInputs):
        '''
        Constructor from Phono3pyInputs class. Takes the Phono3pyInputs object and calculates a specific Brillouin zone property. Specific properties are specified by child classes.

        Parameters
        -------
        inputs : Phono3pyInputs
            input containing all Phono3py input files and tags.
        '''
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
        '''
        Private setter class for interpolation using the Interpolator class
        '''
        from src.Interpolation import Interpolator
        self._interpolator = Interpolator(phonon_property=self)

    def assign_value(self, key, value):
        '''
        Assign value for a given key. The key is either a q-point or a phase-space point depending on the property.
        '''
        self.property[key] = value

    def _shift_q_to_1stBZ(self, qpoint):
        '''
        Internal class that shifts a given q-point back to the 1st Brillouin zone.

        Parameters
        -------
        qpoint : List
            q-point vector in reduced units that may or may not be outside the 1st Brillouin zone.

        Returns
        ------
        The shifted q-point vector.
        '''
        shifted_qpoint = []
        for q in qpoint:
            if q <= -0.5:
                q += 1.
            elif q > 0.5:
                q -= 1.
            shifted_qpoint.append(q)
        return np.array(shifted_qpoint)

    def set_key(self, qpoint, band_index):
        '''
        Set the dictionary key given a q-point and phonon band-index.

        Parameters
        -------
        qpoint : List
            q-point vector labeling the property.
        band_index : int
            Integer index specifying the phonon mode index.

        Returns
        -------
        A vector ready to be used as a dictionary key.
        '''
        key = []
        thresh = 1e-3
        for q, s, m in zip(qpoint, self.shift, self.mesh):
            key_index = (q - s) * m
            rounded_key_index = np.round(key_index).astype(int)
            if np.abs(rounded_key_index - key_index) < thresh:
                key.append(np.round(key_index).astype(int))
            else:
                key.append(key_index)
        key.append(band_index)
        return tuple(key)

    def get_property_value(self, qpoint, band_index):
        '''
        Get a property value given a q-point and band_index. The qpoint and band_index values are used to create a dict key, which then accesses the internal dictionary to extract the value.

        Parameters
        -------
        qpoint : List
            q-point vector labeling the property value.
        band_index : int
            Integer index specifying the phonon mode index.
        '''
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
                args = np.array([list(qpoint) + [f] for f in self.freqs])
                # pass the args into the interpolator
                return self._interpolator.interpolate(band_index, *args)


    def set_property_dict(self):
        '''
        Construct the dictionary object containing all the property data. This module changes depending on the property.
        '''
        pass


#from numba.experimental import jitclass
#@jitclass
# Keep numba part commented for now.
'''
from numba import jit
@jit
def _init_BZ_numba(mesh: np.array, mapping: np.ndarray, grid: np.ndarray):
    new_grid = {}
    for v, k in enumerate(grid):
        new_grid[tuple(k / mesh)] = v

    #grid = {tuple(k / mesh): v for (v, k) in enumerate(grid)} # return
    inverse_grid = {}
    for v, k in enumerate(grid):
        inverse_grid[v] = tuple(k / mesh)
    #inverse_grid = {v: tuple(k / mesh) for (v, k) in enumerate(new_grid)} # return
    temp_irr_BZ_gridpoints = np.unique(mapping)
    irr_BZ_gridpoints = {}
    for v, k in enumerate(temp_irr_BZ_gridpoints):
        irr_BZ_gridpoints[k] = v
    #irr_BZ_gridpoints = {k: v for v, k in enumerate(irr_BZ_gridpoints)} # return

    # Create a dict of irreducible q-points; key is a q-point in full grid, and value is the irreducible q-point
    irr_BZ_qpoints = {}
    for k, gp in new_grid.items():
        irr_BZ_qpoints[k] = inverse_grid[mapping[gp]]
    #irr_BZ_qpoints = {k: inverse_grid[mapping[gp]] for k, gp in new_grid.items()} # return
    # Create dictionary of weights for each irred q-point
    weights = {}
    for irred_gp in temp_irr_BZ_gridpoints:
        weights[inverse_grid[irred_gp]] = list(mapping).count(irred_gp)
    #weights = {inverse_grid[irred_gp]: list(mapping).count(irred_gp) # return
    #                for irred_gp in temp_irr_BZ_gridpoints}
    return (new_grid, inverse_grid, irr_BZ_gridpoints, irr_BZ_qpoints, weights)
'''

class BrillouinZone(MPGrid):
    """
    Abstract class for organizing all annoying gridpoints and mappings within Phono3py. Built from the MPGrid super class.
    """
    def __init__(self, poscar='POSCAR', temperature=0., **kwargs):
        '''
        Constructor from vasp POSCAR, temperature and MPGrid keyword args (mesh and shift).

        Parameters
        -------
        poscar : str
            VASP POSCAR file.
        temperature : float
            Temperature of material.
        mesh : List
            List of integers specifying a Gamma-centered Monkhorst-Pack grid
        shift : List
            List of floats specifying the shift from a Gamma-centered grid.
        '''
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
        # TODO think what to do with numba in this section
        # numba init BZ for large BZ's
        #self.mapping, grid = get_ir_reciprocal_mesh(mesh=self.mesh, cell=self.cell)
        #print(type(self.mapping))
        #print(type(grid))
        #data = _init_BZ_numba(np.array(self.mesh), self.mapping, grid)
        #self.grid, self.inverse_grid, self.irr_BZ_gridpoints, self.irr_BZ_qpoints, self.weights = data


    def _init_BZ(self):
        '''
        Private class to initialize the Brillouin zone.
        '''
        self.mapping, grid = get_ir_reciprocal_mesh(mesh=self.mesh, cell=self.cell)
        self.grid = {tuple(k / self.mesh): v for (v, k) in enumerate(grid)}
        self.inverse_grid = {v: tuple(k / self.mesh) for (v, k) in enumerate(grid)}
        irr_BZ_gridpoints = np.unique(self.mapping)
        self.irr_BZ_gridpoints = {k: v for v, k in enumerate(irr_BZ_gridpoints)}

        # Create a dict of irreducible q-points; key is a q-point in full grid, and value is the irreducible q-point
        self.irr_BZ_qpoints = {k: self.get_qpoint(self.mapping[gp]) for k, gp in self.grid.items()}
        # Create dictionary of weights for each irred q-point
        symm_qpoints_list = [[] for _ in irr_BZ_gridpoints]
        weight_list = np.zeros(len(irr_BZ_gridpoints))
        for index, irred_gp in enumerate(self.mapping):
            symm_qpoints_list[self.irr_BZ_gridpoints[irred_gp]].append(self.qpoints[index])
            weight_list[self.irr_BZ_gridpoints[irred_gp]] += 1
        self.symm_qpoints = {}
        self.weights = {}
        for irr_gp, symm_qpoints, weight in zip(irr_BZ_gridpoints, symm_qpoints_list, weight_list):
            self.symm_qpoints[tuple(self.get_qpoint(irr_gp))] = symm_qpoints
            self.weights[tuple(self.get_qpoint(irr_gp))] = weight

        #self.weights = {self.get_qpoint(irred_gp): list(self.mapping).count(irred_gp)
        #                for irred_gp in irr_BZ_gridpoints}

    def get_symmetrically_equiv_qpoints(self, qpoint):
        '''
        Get the symmetrically equivalent q-points given a specific q-point.

        Parameters
        -------
        qpoint : List
            Vector of a q-point in reduced coordinates.

        Returns
        -------
        List of q-point vectors that are symmetrically equivalent to the original q-point.
        '''
        gp = self.get_gridpoint(qpoint)
        irr_gp = self.mapping[gp]
        qpoints = []
        for index, tmp_gp in enumerate(self.mapping):
            if irr_gp == tmp_gp:
                qpoints.append(self.qpoints[index])
        return qpoints

    def get_gridpoint(self, qpoint):
        '''
        Get a grid-point given a q-point. Gridpoint is used to properly access Phono3py objects.

        Parameters
        ------
        qpoint : List
            Vector of a q-point in reduced coordinates

        Returns
        -------
        The gridpoint associated with the passed q-point (gridpoint is an integer).
        '''
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
        '''
        Get qpoint from a gridpoint.

        Parameters
        -------
        gridpoint : int
            Integer specifying the gridpoint used in Phono3py objects.

        Returns
        -------
        q-point vector associated with gridpoint.
        '''
        return self.inverse_grid[gridpoint]

    def shift_q_to_1stBZ(self, qpoint):
        '''
        Shift q-point to 1st Brillouin zone.

        Parameters
        -------
        qpoint : List
            Vector of a q-point in reduced coordinates
        '''
        shifted_qpoint = []
        for q in qpoint:
            sq = q % 1.
            if sq <= -0.5:
                sq += 1.
            elif sq > 0.5:
                sq -= 1.
            shifted_qpoint.append(sq)
        return np.array(shifted_qpoint)

class Phono3pyManager:
    '''
    Class that manages all relevant functions of the Phono3py API, and connects these functions with the other classes in this mpdsf. This manager has not been checked for Phono3py versions >2.0; this will be looked into in the future.
    '''
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
        self.phono3py.init_phph_interaction(nac_q_direction=[1, 0, 0])
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
        '''
        Calculate phonon data using Phono3py. Store the frequencies, eigenvectors and qpoints.
        '''
        #self.phono3py.run_phonon_solver()
        self.bands, self.eigvecs, self.qpoints = self.phono3py.get_phonon_data()
        self.fix_phonon_data()

    def fix_phonon_data(self):
        '''
        Fix phonon data to remove the redundant q-points and shift the q-points back to 1st Brillouin zone.
        '''
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
        '''
        Move any q-point outside the first Brillouin zone back into the first zone
        '''
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
        '''
        Imaginary self energy objects are stored in phono3py object in a weird way.

        The object is a list of a list of ndarrays.
        
        First index = grid-points, which are "addresses" of the irreducible q-points in the Brillouin zone. At the surface these are meaningless, but the grid points are given by: np.unique(mapping). The actual q-points are stored in grid with a 1-to-1 correspondence to mapping
        
        Second index = sigma values, there is only one sigma=None in default tetrahedron method
        
        Third index is the nparray, which is arranged as: (temperatures, frequency points, band index)
        '''
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
        '''
        Private function for getting imaginary self energy functions from Phono3py after specifying a q-point.

        Note that this returns the imaginary self energy function.
        Does not return the self energy evaluated at the phonon frequency.
        As a result, a tuple is returned with the frequency points, and self energy.

        Parameters
        -------
        gridpoint : int
            Integer specifying the Phono3py gridpoint.

        Returns
        -------
        Imaginary self energy function
        '''
        irr_gp = self._brillouinzone.mapping[gridpoint]
        grid_index = self._brillouinzone.irr_BZ_gridpoints[irr_gp]
        return self.manager.phono3py._frequency_points, self.manager.phono3py._gammas[grid_index][0][0, :, :]

    def get_imag_self_energies_at_q(self, qpoint):
        '''
        Get imaginary self energy functions at a given q-point (instead of a gridpoint).

        Parameters
        -------
        qpoint : List
            q-point vector in reduced coordinates

        Returns
        ------
        Imaginary self energy function.
        '''
        gridpoint = self._brillouinzone.grid[tuple(qpoint)]
        _, ises = self._get_imag_self_energies_from_gp(gridpoint)
        return ises

    def set_property_dict(self):
        '''
        Calculate the property dictionary for the imaginary self energy function.
        '''
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
    '''
    Extension of BrillouinZoneProperty to calculate phonon eigenvalues, and store/access them simply in one place. This is a scalar valued property.
    '''
    def __init__(self, inputs: Phono3pyInputs):
        '''
        Constructor from Phono3py inputs object.

        Parameters
        -------
        inputs : Phono3pyInputs
            Class containing all Phono3py related input files and tags.
        '''
        super().__init__(inputs)
        self.eigenvalues = {}
        self.set_property_dict()

    def set_property_dict(self):
        '''
        Calculate the property dictionary for phonon eigenvalues.
        '''
        self.manager.set_phonons()
        for i, q in enumerate(self._brillouinzone.qpoints):
            for band_index, eig in enumerate(self.manager.bands[i, :]):
                #key = tuple(q)
                #key += (band_index,)
                key = self.set_key(q, band_index)
                self.property_dict[key] = eig


class PhononEigenvectors(BrillouinZoneProperty):
    '''
    Extension of BrillouinZoneProperty to calculate phonon eigenvectors, and store/access them in one place. This is a *vector* valued property.
    '''
    def __init__(self, inputs: Phono3pyInputs):
        '''
        Constructor from Phono3py inputs object.

        Parameters
        -------
        inputs : Phono3pyInputs
            Class containing all Phono3py related input files and tags.
        '''
        super().__init__(inputs)
        self.eigenvectors = {}
        self.set_property_dict()

    def set_property_dict(self):
        '''
        Calculate the property dictionary for phonon eigenvectors.
        IMPORTANT NOTE: I believe the way Phono3py stores it's eigenvectors is in a matrix of shape:
            (num qpts, num bands, 3 * num_atoms)
        
        This is hard to test because the shape of the matrix at each q-point is square; (num bands) = (3 * num atoms)
        
        However, in the Phono3py source code, the eigenvectors are determined via a call to linalg.eigh acting on the dynamical matrix. This call returns the eigenvectors as column vectors in a square matrix such that the second index is eigenvector index, and the first index corresponds to the eigenvector components
        '''
        self.manager.set_phonons()
        for i, q in enumerate(self._brillouinzone.qpoints):
            for band_index, eigvec in enumerate(self.manager.eigvecs[i].T):
                # impossible to know which index is actually the eigenvector b/c it is a square matrix
                key = self.set_key(q, band_index)
                self.property_dict[key] = eigvec


class GroupVelocities(BrillouinZoneProperty):
    '''
    Extension of BrillouinZoneProperty to calculate phonon group velocities, and store/access them in one place. This is a *vector* valued property. Units are Angstroms / ps.
    '''
    def __init__(self, inputs: Phono3pyInputs):
        '''
        Constructor from Phono3py inputs object.

        Parameters
        -------
        inputs : Phono3pyInputs
            Class containing all Phono3py related input files and tags.
        '''
        super().__init__(inputs)
        self.group_velocities = {}
        self.set_property_dict()

    def set_property_dict(self):
        '''
        Calculate the property dictionary for group velocities using Phono3py API, and store internally in class. The unit is in Angstroms / ps.
        '''
        self.manager.phono3py.init_phph_interaction()
        group_vels = GroupVelocity(self.manager.phono3py.dynamical_matrix)

        group_vels.run(self._brillouinzone.qpoints)

        for q, gvs_at_q in zip(self._brillouinzone.qpoints, group_vels.group_velocities):
            for branch_index, gv in enumerate(gvs_at_q):
                key = self.set_key(qpoint=q, band_index=branch_index)
                self.property_dict[key] = gv



class IsotopicImagSelfEnergy(BrillouinZoneProperty):
    def __init__(self, inputs: Phono3pyInputs):
        '''
        Constructor from Phono3py inputs object.

        Parameters
        -------
        inputs : Phono3pyInputs
            Class containing all Phono3py related input files and tags.
        '''
        super().__init__(inputs)
        self.ise = None
        self.units = 'THz = 1 / (4pi * tau)'
        self.imag_self_energy = {}
        self.set_self_energies()
        self.set_property_dict()

    def set_self_energies(self):
        '''
        Set isotopic self energies using Phono3pyIsotope class. The self energies are stored in self.manager.isotope.gamma with units of THz.
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
        '''
        Calculate the property dictionary for the isotopic self energy for each q-point in the MP grid.
        '''
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
    Extension of BrillouinZoneProperty to calculate the Gamma parameter which is related to the inverse phonon lifetime. The relationship in Phono3py units is lifetime=1/(4*pi*Gamma). This class is similar to ImaginarySelfEnergy, but evaluates the ISE at the phonon frequency only
    '''
    def __init__(self, inputs: Phono3pyInputs, f_min=1e-3):
        '''
        Constructor from Phono3py inputs object.

        Parameters
        -------
        inputs : Phono3pyInputs
            Class containing all Phono3py related input files and tags.
        f_min : float
            Minimum frequency. Phonons with frequencies below this threshold are ignored in any analysis and set to 0.
        '''
        super().__init__(inputs)
        self.ise = ImaginarySelfEnergy(inputs)
        self.eigs = PhononEigenvalues(inputs)
        self.f_min = f_min
        self.set_property_dict()

    def set_property_dict(self):
        '''
        Calculate the property dictionary for the Gamma functions.
        '''
        for key, ise_at_key in self.ise.property_dict.items():
            eig_at_key = self.eigs.property_dict[key]
            if eig_at_key < self.f_min:
                self.property_dict[key] = 0.
                continue
            interp = interp1d(self.ise.freqs, ise_at_key)
            self.property_dict[key] = interp(eig_at_key)
