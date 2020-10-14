import numpy as np
from scipy.interpolate import LinearNDInterpolator

class Interpolator:
    def __init__(self,
                 band=None,
                 gamma=None,
                 qpoints=None,
                 freqs=None,
                 kind='linear'):
        """
        Interpolator is an object that interpolates either w(kx, ky, kz) or Gamma(w, kx, ky, kz) to improve BZ sampling
        :param band: ndarray of band frequencies at specified qpoints
        :param gamma: ndarray of imaginary self energies at specified frequency and qpoint pairs
        :param qpoints: list of qpoints corresponding to the band frequencies or Gamma function
        :param freqs: list of frequencies used in the Gamma function
        :param kind: specifies the type of interpolator to use; default is linear because it is most stable
        """
        self.band = band
        self.gamma = gamma
        self.qpoints = qpoints
        self.freqs = freqs
        self.kind = kind
    def set_band(self, band):
        self.band = band
    def set_freqs(self, freqs):
        self.freqs = freqs
    def set_interpolator(self, kind='linear'):
        if self.freqs is not None:
            # Assume the interpolator is for the gamma function
            phasespace_points = self.get_phasespace_points()
            gamma_conv = self.get_converted_gamma_for_interp()
            self.interpolator(LinearNDInterpolator(phasespace_points, gamma_conv))
        else:
            # Assume the interpolator is for the band function
            phasespace_points = self.qpoints
            self.interpolator(LinearNDInterpolator(phasespace_points, self.band))
    def get_phasespace_points(self):
        phasespace_points = np.array([list(q) + freq for q in self.qpoints for freq in self.freqs])
        return phasespace_points
    def get_converted_gamma_for_interp(self):
        return np.array(self.gamma).reshape(-1, order='C')
    def interpolate(self, *args):
        return self.interpolator(args)

class Regridder:
    def __init__(self, interpolable_object,
                 mesh=None, freqs=None):
        self.interpolator = Interpolator(interpolable_object)
        self.mesh = mesh
        self.freqs = freqs

        # set Phase space from mesh and freqs
        self.phase_space = PhaseSpace(freqs=self.freqs, mesh=self.mesh)
    def regrid(self, mesh=None):
        if mesh is not None:
            self.mesh = mesh
            self.phase_space = PhaseSpace(freqs=self.freqs,
                                          mesh=self.mesh)
        # begin the regridding!
        return np.array([self.interpolator(ps_point) for ps_point in self.phase_space])

class MPGrid:
    def __init__(self, mesh, shift=None):
        self.mesh = mesh
        if shift is not None:
            self.shift = shift
        else:
            self.shift = [0., 0., 0.]
        self.qpoints = None

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

class PhaseSpace(MPGrid):
    def __init__(self, freqs=None, **kwargs):
        super(**kwargs)
        self.freqs = freqs
        self.phase_space = None

        # set phase_space from freqs and qpoints from super()
        if freqs is not None:
            self.set_phase_space()
        else:
            # if freqs is not given, then PhaseSpace object acts exactly like an MPGrid object
            self.phase_space = self.qpoints
    def set_phase_space(self):
        self.phase_space = np.array([list(q) + freq for q in self.qpoints for freq in self.freqs])


class ImaginarySelfEnergy:
    def __init__(self):
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

class Band:
    def __init__(self):
        pass