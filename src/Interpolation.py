import numpy as np
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from src.utils import BrillouinZoneProperty, PhaseSpace

class Interpolator:
    def __init__(self, phonon_property: BrillouinZoneProperty, reg_grid_flag=True):
                 #band=None,
                 #gamma=None,
                 #qpoints=None,
                 #freqs=None,
                 #kind='linear'):
        """
        Interpolator is an object that interpolates either w(kx, ky, kz) or Gamma(w, kx, ky, kz) to improve BZ sampling
        :param band: ndarray of band frequencies at specified qpoints
        :param gamma: ndarray of imaginary self energies at specified frequency and qpoint pairs
        :param qpoints: list of qpoints corresponding to the band frequencies or Gamma function
        :param freqs: list of frequencies used in the Gamma function
        :param kind: specifies the type of interpolator to use; default is linear because it is most stable
        """

        self.property = phonon_property
        self.phase_space = PhaseSpace(freqs=self.property.freqs,
                                      mesh=self.property.mesh,
                                      shift=self.property.shift)
        self.interpolators = None
        self._vector_flag = None
        self._vector_length = None
        self._reg_grid_flag = reg_grid_flag
        #self.band = band
        #self.gamma = gamma
        #self.qpoints = qpoints
        #self.freqs = freqs
        #self.kind = kind
    """
    def set_band(self, band):
        self.band = band
    def set_freqs(self, freqs):
        self.freqs = freqs
    """
    def check_scalar(self):
        # Need to check if the property is a scalar or vector quantity
        check_qpt = self.phase_space.qpoints[0]
        check_val = self.property.get_property_value(check_qpt, band_index=0)
        # I think the safest way to do this is to create an ndarray and look at it's shape
        check_arr = np.array([check_val])
        if len(check_arr.shape) == 1:
            return True
        else:
            return False

    def set_interpolators(self, num_bands=None, vector_flag=None, vector_length=None):
        """
        This module sets all interpolators for the interpolable function. For a scalar function, like phonon
        eigenvalues, the operation is simple, as it builds a dict of interpolators with the band indices as keys. For a
        vector quantity, like phonon eigenvectors, it builds a dict of interpolators for each component and band index,
        and the tuple (band_index, vector_index) is the key. For a frequency dependent function, like the imaginary
        self energy, the result is the same as the scalar case, where a dict is built with band indices as keys.
        :param vector_flag: Boolean describing whether or not the object is a vector
        :param vector_length: Integer input that allows user to define the dimensionality of the interpolated vector.
                              If vector_flag is True and no length is given, then the length will be assumed to be
                              3 * num_atoms
        :return: None
        """
        if num_bands is None:
            num_bands = 3 * len(self.property.manager.phono3py.primitive.masses)
        if vector_flag is None:
            vector_flag = not self.check_scalar()
        if self.property.freqs is not None:
            self._vector_flag = False
        else:
            self._vector_flag = vector_flag

        if vector_length is None and self._vector_flag:
            # In this case, assume that vector length is equal to number of eigenvectors
            vector_length = num_bands
        self._vector_length = vector_length
        # Initialize dict
        self.interpolators = {}
        for i in range(num_bands):
            if self._vector_flag:
                for j in range(vector_length):
                    self.interpolators[(i, j)] = self.set_interpolator_at_band(band_index=i, vector_index=j)
            else:
                self.interpolators[i] = self.set_interpolator_at_band(band_index=i)

    def set_interpolator_at_band(self, band_index, vector_index=None):
        if self._reg_grid_flag:
            if self.property.freqs is None:
                phasespace_points = self.phase_space.get_padded_qpoints()
            else:
                phasespace_points = self.phase_space.get_padded_phase_space()
            points = []
            for i in range(3):
                min_q = min(phasespace_points[:, i])
                max_q = max(phasespace_points[:, i])
                num_q = np.round((max_q - min_q) * self.phase_space.mesh[i]).astype(int) + 1
                points.append(np.linspace(min_q, max_q, num_q))
            if self.property.freqs is None:
                converted_property = np.empty([len(points[0]), len(points[1]), len(points[2])])
            else:
                points.append(self.property.freqs)
                converted_property = np.empty([len(points[0]),
                                               len(points[1]),
                                               len(points[2]),
                                               len(self.property.freqs)])
            for i in range(len(points[0])):
                for j in range(len(points[1])):
                    for k in range(len(points[2])):
                        qpt = [points[0][i], points[1][j], points[2][k]]
                        property_value = self.property.get_property_value(qpt, band_index)
                        if vector_index is not None:
                            converted_property[i, j, k] = property_value[vector_index]
                        elif self.property.freqs is not None:
                            converted_property[i, j, k, :] = property_value
                        else:
                            converted_property[i, j, k] = property_value
            return RegularGridInterpolator(tuple(points), converted_property)
        else:
            if self.property.freqs is not None:
                # Assume the interpolator is for the freq dep function
                phasespace_points = self.phase_space.get_padded_phase_space()

                # Need to convert the property
                converted_property = self.convert_freq_dep_property(band_index)
                return LinearNDInterpolator(phasespace_points, converted_property)
            else:
                # Assume the interpolator is for the band function
                phasespace_points = self.phase_space.get_padded_qpoints()

                # Convert scalar or vector quantity
                converted_property = self.convert_property(band_index=band_index, vector_index=vector_index)
                return LinearNDInterpolator(phasespace_points, converted_property)

    def convert_freq_dep_property(self, band_index):
        data = []
        for q in self.phase_space.get_padded_qpoints():
            prop_func = self.property.get_property_value(q, band_index)
            # frequencies change the fastest in the list of phase-space values, so a simple append works here
            data += list(prop_func)
        return data

    def convert_property(self, band_index, vector_index=None, padded_flag=True):
        # Set padded_flag to true because that ensures proper action of an interpolator that is periodic
        data = []
        if padded_flag:
            qpts = self.phase_space.get_padded_qpoints()
        else:
            qpts = self.phase_space.qpoints

        for q in qpts:
            q_1stBZ = self.property._shift_q_to_1stBZ(q)
            if vector_index is None:
                # Scalar mode
                prop_func = self.property.get_property_value(q_1stBZ, band_index)
            else:
                prop_func = self.property.get_property_value(q_1stBZ, band_index)[vector_index]
            # frequencies change the fastest in the list of phase-space values, so a simple append works here
            data.append(prop_func)
        return data

    def interpolate(self, band_index, *args):
        """
        Interpolate the function for the desired band and vector index if applicable
        :param band_index:
        :param vector_index:
        :param args: arguments for interpolator in a list [qx, qy, qz] or [qx, qy, qz, w] for freq dep properties
        :return:
        """
        args = list(args)
        if len(np.array(args).shape) > 1:
            new_args = []
            for arg in args:
                new_args.append(list(self.property._shift_q_to_1stBZ(arg[:3])) + list(arg[3:]))
            args = [new_args]
        else:
            args[:3] = self.property._shift_q_to_1stBZ(args[:3])
        if self.interpolators is None:
            self.set_interpolators()
        if not self._vector_flag:
            if self._reg_grid_flag:
                return self.interpolators[band_index](args)[0]
            else:
                return self.interpolators[band_index](*args)
        else:
            if self._reg_grid_flag:
                interpolator_list = [self.interpolators[(band_index, i)](args)[0] for i in range(self._vector_length)]
            else:
                interpolator_list = [self.interpolators[(band_index, i)](*args) for i in range(self._vector_length)]
            return interpolator_list

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

