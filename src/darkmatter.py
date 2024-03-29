import numpy as np
from scipy import special
import numba as nb
import h5py
import json
from yaml import load
from yaml import CLoader as Loader
from src.units import *
from src.compute_Sqw import DynamicStructureFactor

# Conversion factors pulled out of functions
# Density conversion
density_conversion = to_unit_sys_mag(1 * grm / cmet ** 3, 'NateV')
# cross-sec conversion
cross_sec_conversion = to_unit_sys_mag(10 ** (-40) * cmet ** 2, 'NateV')
# exposure conversion
exposure_conversion = to_unit_sys_mag(kgrm * year, 'NateV')
# rate conversion
rate_conversion = to_unit_mag(1.0 * eV ** (-2), cmet ** 2)

## Define all relevant constants

# DM density
rho_chi = to_unit_sys_mag(0.4*GeV/cmet**3, 'NateV')

# fine structure constant
alpha_EM = 1/137
# electron mass
m_elec = to_unit_sys_mag(511*keV, 'NateV')
# proton mass
m_proton = to_unit_sys_mag(938*MeV, 'NateV')

# velocity dispersion
v0 = to_unit_sys_mag(230*kmet/sec, 'NateV')
# Earth velocity
vE = to_unit_sys_mag(240*kmet/sec, 'NateV')
# galactic escape velocity
vEsc = to_unit_sys_mag(600*kmet/sec, 'NateV')
# velocity distribution normalization constant
N0 = np.pi**(3/2)*v0**2*(v0*special.erf(vEsc/v0) - (2/np.sqrt(np.pi))*vEsc*np.exp(-vEsc**2/v0**2))

# direction of the Earth velocity relative to the DM wind
thetaE = (np.pi/180)*42


class ReachCalculator:
    '''
    Object that is currently implemented to take an hdf5 file created from the multiphonon_dsf.py code
    and create a reach curve for a default set of masses to produce publication quality data in an 
    efficient and scalable way.

    Inputs:
    ------
        - sqw: NOT IMPLEMENTED YET; DO NOT USE
        - hdf5_file: filename for hdf5 file created by multiphonon_dsf.py
        - equiv_flag: Boolean that describes whether Brillouin zone was folded into the irreducible part
                      (Eventually this will get removed as this function can be inferred from the hdf5 
                      file)

    Methods:
    -----
        - calculate_reach:
            Inputs: 
                - dm_masses: (optional) array of DM masses in eV
            Returns:
                - reach: An array of reach values for each DM mass
    '''
    def __init__(self, sqw: DynamicStructureFactor=None, hdf5_file=None, equiv_flag=True):
        self._sqw = sqw
        self._hdf5_file = hdf5_file
        self._equiv_flag = equiv_flag
        
        self.data_dict = None
        self.reach = None
        if self._sqw is not None:
            self.load_from_dsf()
        else:
            self.load_from_file()
    ## Import dynamic structure factor from compute_sqw

    def get_q_grid(self,
                   filename=None,
                   qpts_key='q-points',
                   reclat_key='reclat',
                   symm_points_key='equivalent q-points'):
        """
        Return a list of q vectors. Output array should have dim ( N_q, 3 ), where N_q is the number of q
        points.

        q_vectors should have units : eV

        Parameters
        -------
        filename : str
            Name of hdf5 file
        qpts_key : str
            Either normal q-points key ('q-points') or scaling key ('scaling q-points')
        reclat_key : str
            Key for reciprocal lattice matrix
        symm_points_key : str
            key for symmetrically equivalent q-points dictionary.

        Returns
        -------
        q_grid matrix
        """
        if filename is not None:
            f = h5py.File(filename, 'r')
            if symm_points_key is not None:
                if symm_points_key == 'equivalent scaling_qpoints' and symm_points_key not in f.keys():
                    symm_points_key = 'scaling equivalent_q-points'
                if symm_points_key == 'scaling equivalent_q-points' and symm_points_key not in f.keys():
                    symm_points_key = 'equivalent scaling_q-points'
                #s = str(np.array(f[symm_points_key]))
                #if s[0] != '-':
                #    s = s[2:-1]
                # Remove \\n parts of string literal
                #s_array = s.split('\\n')
                # Add \n parts back in
                #s = '\n'.join(s_array)
                #symm_points = load(s, Loader=Loader)
                symm_points_dict = json.loads(f[symm_points_key].asstr()[()])
                #symm_points = load(str(np.array(f[symm_points_key])), Loader=Loader)
                q_grid = []
                qpts = np.array(f[qpts_key])
                if len(qpts) == 0:
                    return qpts
                for q in qpts:
                    key = '{:.8f},{:.8f},{:.8f}'.format(*q)
                    symm_points = symm_points_dict[key]
                    q_grid += list(symm_points)
                q_grid = np.array(q_grid)
            else:
                q_grid = np.array(f[qpts_key])
            reclat = np.array(f[reclat_key])
        elif self._sqw is not None:
            reclat = self._sqw.rec_lat * 2 * np.pi
            if self._sqw.fold_BZ:
                irr_gps = np.unique(self._sqw._brillouinzone.mapping)
                symm_points = []
                for gps in irr_gps:
                    q = self._sqw._brillouinzone.qpoints[gps]
                    symm_points.append(list(np.array(self._sqw._brillouinzone.symm_qpoints[tuple(q)]) *
                                            np.array(self._sqw.meshG)))
                q_grid = []
                for q in symm_points:
                    q_grid += list(q)
                q_grid = np.array(q_grid)
            else:
                q_grid = np.array(self._sqw.qpoints)
        else:
            print('One needs to include either a file or DynamicStructureFactor object for this to work. '
                  'Code will exit with error.')
        unit_conv = 1 / to_unit_sys_mag(Ang, 'NateV')
        return np.dot(q_grid, reclat) * unit_conv

    #TODO finish integrating this code with DynamicStructureFactor code
    def get_omega_grid(self, filename, freq_key='frequencies'):
        """
        Return a list of omega values. Output array should have dim ( N_w ), where N_w is the number of omega
        points.

        omega's should have units : eV

        Parameters
        -------
        filename : str
            Name of HDF5 file.
        freq_key : str
            Dict key for phonon frequencies.  (optional)

        Returns
        -------
        Array of phonon frequencies
        """
        frequencies = np.array(h5py.File(filename, 'r')[freq_key])
        unit_conv = to_unit_sys_mag(THz, 'NateV') * 2 * np.pi
        return frequencies * unit_conv


    def get_jac_q(self, filename, jacq_key='dxdydz', weight_key='weights', symm_points_key='equivalent q-points'):
        """
        Return the jacobian for the q grid integration. Output array should have dim ( N_q ),
        where N_q is the number of q points.

        jacobian should have units of eV^3

        Parameters
        -------
        filename : str
            Name of HDF5 file.
        jacq_key : str
            Key for q-point jacobian.
        weight_key : str
            Key for list of weights
        symm_points_key : str
            Key for symmetrically equivalent q-points

        Returns
        -------
        List of q-point jacobians for each q-point.
        """
        weights = np.array(h5py.File(filename, 'r')[weight_key])
        jac_q = np.array(h5py.File(filename, 'r')[jacq_key])
        # num_qpts = len(np.array(h5py.File(filename, 'r')[qpts_key]))
        unit_conv = to_unit_sys_mag(Ang, 'NateV') ** -3
        # return 1. * unit_conv * np.ones(num_freqs)
        if symm_points_key is None:
            return jac_q * unit_conv * weights / (8 * np.pi ** 3)
        else:
            return jac_q * unit_conv * np.ones(int(sum(weights))) / (8 * np.pi ** 3)


    def get_jac_omega(self, filename, jacw_key='delta_w', freq_key='frequencies'):
        """
        Return the jacobian for the w grid integration. Output array should have dim ( N_w ),
        where N_w is the number of omega points.

        jacobian should have units of eV

        Parameters
        -------
        filename : str
            Name of HDF5 file.
        jacw_key : str
            Key for omega jacobian (also called dw).
        freq_key : str
            Key for phonon frequencies.

        Returns
        -------
        List of omega jacobians.
        """
        jac_omega = np.array(h5py.File(filename, 'r')[jacw_key])
        num_freqs = len(np.array(h5py.File(filename, 'r')[freq_key]))
        unit_conv = to_unit_sys_mag(THz, 'NateV') * 2 * np.pi
        return jac_omega * unit_conv * np.ones(num_freqs)


    def get_dyn_structure_factor(self, filename, sqw_key='sqw',
                                 jacq_key='dxdydz',
                                 reclat_key='reclat',
                                 symm_points_key='equivalent q-points',
                                 qpts_key='q-points'):
        """
        Return the dynamic structure factor. Output array should have dim ( N_q, N_w ),
        where N_q is the number of points in the q grid and N_w is the number of omega points.

        dynamic strucure factor should have units of eV^2

        Parameters
        -------
        filename : str
            Name of HDF5 file.
        sqw_key : str
            Key for dynamic structure factor object.
        jacq_key : str
            Key for q-point jacobian
        reclat_key : str
            Key for reciprocal lattice vector matrix
        symm_points_key : str
            Key for symmetrically equivalent q-points
        qpts_key : str
            Key for list of q-points

        Returns
        ------
        Dynamic structure factor matrix.
        """
        f = h5py.File(filename, 'r')
        sqw = np.array(f[sqw_key])
        if len(sqw) == 0:
            return sqw
        if symm_points_key is not None:
            if symm_points_key == 'equivalent scaling_qpoints' and symm_points_key not in f.keys():
                symm_points_key = 'scaling equivalent_q-points'
            if symm_points_key == 'scaling equivalent_q-points' and symm_points_key not in f.keys():
                symm_points_key = 'equivalent scaling_q-points'
            #s = str(np.array(f[symm_points_key]))
            #if s[0] != '-':
            #    s = s[2:-1]
            # Remove \\n parts of string literal
            #s_array = s.split('\\n')
            # Add \n parts back in
            #s = '\n'.join(s_array)
            #symm_points = load(s, Loader=Loader)
            
            #symm_points = np.array(load(str(np.array(f[symm_points_key])), Loader=Loader))
            symm_points_dict = json.loads(f[symm_points_key].asstr()[()])
            unfolded_sqw = []
            qpts = np.array(f[qpts_key])
            for sqw_at_q, q in zip(sqw, qpts):
                key = '{:.8f},{:.8f},{:.8f}'.format(*q)
                qs = symm_points_dict[key]
                for _ in qs:
                    unfolded_sqw.append(sqw_at_q)
            unfolded_sqw = np.array(unfolded_sqw)
            sqw = unfolded_sqw
        if sqw_key == 'scaling_sqw':
            jac_q = 1.
        else:
            jac_q = np.array(f[jacq_key])
        primitive_vol = (2 * np.pi) ** 3 * np.linalg.det(np.array(f[reclat_key])) ** -1
        unit_conv = ((2 * np.pi * to_unit_sys_mag(THz, 'NateV')) ** -1) * (to_unit_sys_mag(Ang, 'NateV') ** -3)
        return 2 * np.pi / primitive_vol * sqw * jac_q * unit_conv


    def get_all_data_dict(self, filename, equiv_flag=True):
        """
        Collection of previous get functions returning an array of the relevant data given the data
        filename

        Parameters
        -------
        filename : str
            Name of HDF5 file.
        equiv_flag : bool
            Boolean flag specifying whether or not symmetrically equivalent objects are in the HDF5 file.
        """
        f = h5py.File(filename, 'r')
        self._density = np.array(f['density'])
        self._max_freq = np.array(f['max freq'])
        if equiv_flag:
            data_dict = {
                "q_grid": list(self.get_q_grid(filename)),
                "omega_grid": list(self.get_omega_grid(filename)),
                "jac_q": list(self.get_jac_q(filename)),
                "jac_omega": list(self.get_jac_omega(filename)),
                "dyn_S": list(self.get_dyn_structure_factor(filename))
            }
        else:
            data_dict = {
                "q_grid": list(self.get_q_grid(filename, symm_points_key=None)),
                "omega_grid": list(self.get_omega_grid(filename)),
                "jac_q": list(self.get_jac_q(filename, symm_points_key=None)),
                "jac_omega": list(self.get_jac_omega(filename)),
                "dyn_S": list(self.get_dyn_structure_factor(filename, symm_points_key=None))
            }
        if 'scaling_sqw' in f.keys():
            print('including scaling data...')
            if equiv_flag:
                symm_points_key = 'equivalent scaling_qpoints'
            else:
                symm_points_key = None
            q_grid = list(self.get_q_grid(filename,
                                          qpts_key='scaling_q-points',
                                          symm_points_key=symm_points_key))

            data_dict["q_grid"] += q_grid
            if len(q_grid) > 0:
                data_dict["jac_q"] += list(self.get_jac_q(filename, jacq_key='scaling_dxdydz',
                                                          weight_key='scaling_weights',
                                                          symm_points_key=symm_points_key))
            data_dict["dyn_S"] += list(self.get_dyn_structure_factor(filename,
                                                                     sqw_key='scaling_sqw',
                                                                     jacq_key='scaling_dxdydz',
                                                                     symm_points_key=symm_points_key,
                                                                     qpts_key='scaling_q-points'
                                                                    ))
        return data_dict

    def load_from_file(self):
        '''
        Load phonon data from HDF5 file.
        '''
        self.data_dict = self.get_all_data_dict(self._hdf5_file, equiv_flag=self._equiv_flag)

    def load_from_dsf(self):
        '''
        Load from DynamicStructureFactor object. Not implemented yet, will quit program if called.
        '''
        raise SystemExit('Not implemented yet... sorry, will happen sometime soon')

    def calculate_reach(self, 
            num_masses=20, 
            min_mass=1e3, 
            max_mass=1e10, 
            threshold=1e-3, 
            t=0,
            dm_masses=None,
            ref='nucleon',
            med='light'):
        '''
        Calculate the dark matter reach (minimal scattering cross-section for 3 events/year into a 1 kg
        block of material) for a specified model and threshold
        
        Parameters
        -------
        num_masses : int 
            number of dark matter masses to calculate reach for.
        min_mass : float
            minimum dark matter mass (in eV) to calculate reach for.
        max_mass : float
            maximum dark matter mass
        threshold : float
            phonon frequency cutoff in which phonons with energy below this cutoff do not contribute to the reach.
        t : float
            time of day in hours
        dm_masses : List
            specify the array of dm_masses directly instead of using (num_masses, min_mass, max_mass) in conjunction with np.logspace
        ref : str
            reference particle used for determining dark matter interaction model. Implemented 'nucleon' and 'electron'
        med : str
            mediator tag determining dark matter model. Implemented 'light' and 'heavy'

        Returns
        -------
        List of reach values for each dark matter mass specified.
        '''
        density = self._density
        dm_masses = np.logspace(np.log10(min_mass), np.log10(max_mass), num_masses)
        reach = np.empty_like(dm_masses)
        for i, mass in enumerate(dm_masses):
            #print('i =', i, 'mass =', mass)
            reach[i] = ph_dd_cross_sec_constraint_numba(mass, density, t,
                                                      np.array(self.data_dict['q_grid']),
                                                      np.array(self.data_dict['omega_grid']),
                                                      np.array(self.data_dict['jac_q']),
                                                      np.array(self.data_dict['jac_omega']),
                                                      np.array(self.data_dict['dyn_S']),
                                                      ref=ref,
                                                      med=med,
                                                      threshold=threshold)
        self.reach = reach
        return reach

@nb.jit
def ph_dd_rate_numba(m_chi, rho_target, t,
                     q_grid, omega_grid, jac_q, jac_omega,
                     dyn_structure_factor,
                     cross_sec=1.,
                     ref="electron",
                     med="light",
                     rate=True,
                     threshold=1e-3):
    """
    Compute the direct detection rate given the dynamic structure factor. N_q is number of q points
    and N_w is the number of omega points

    Integrals are discretized as:

        int d^3q    -> sum_q jac_q
        int d omega -> sum_omega jac_omega

    Inputs : m_chi      - DM mass              units : eV
             rho_target - target density       units : gm/cmet^3
             t          - time                 units : hr

             ref        - cross section to reference to, either 'electron' or 'nucleon'
             med        - type of mediator, either 'heavy' or 'light'

             q_grid     - ( N_q, 3 ) list of momentum transfer vectors  units : eV
             omega_grid - ( N_w ) list of energy transfers              units : eV
             jac_q      - ( N_q ) jacobian of q point                   units : eV^3
             jac_oemga  - ( N_w ) jacobian of omega point               units : eV

             dyn_structure_factor - ( N_q, N_w ) dynamic structure factor    units : eV^2


     Outputs : direct detection rate per kg-year  units : dimensionless

    """
    # problem
    # rho_T = to_unit_sys_mag(rho_target, "NateV")
    rho_T = density_conversion * rho_target
    if rate:
        # problem
        # sigma = to_unit_sys_mag(cross_sec, "NateV")
        sigma = cross_sec_conversion
    else:
        sigma = 1

    if ref == 'electron':
        # mu can be numba-fied
        red_mass = mu_numba(m_elec, m_chi)
    elif ref == 'nucleon':
        red_mass = mu_numba(m_proton, m_chi)

    # allowed by numba
    overall_const = 0.5 * sigma * (rho_chi / rho_T) * (red_mass ** 2 * m_chi) ** (-1)

    rate = 0

    for q, q_vec in enumerate(q_grid):
        q_mag = np.linalg.norm(q_vec)
        if q_mag == 0:
            continue
        for w, omega in enumerate(omega_grid):

            if omega >= threshold:
                # critical that this is allowed
                rate += jac_q[q] * jac_omega[w] * kinematic_function_numba(q_vec, omega, t, m_chi) * \
                        med_form_factor_sq_numba(q_vec, m_chi, ref=ref, med=med) * \
                        dyn_structure_factor[q, w]

    return overall_const * rate


@nb.jit
def ph_dd_cross_sec_constraint_numba(m_chi, rho_target, t,
                                     q_grid, omega_grid, jac_q, jac_omega,
                                     dyn_structure_factor,
                                     n_cl=3,
                                     exposure=1.,
                                     ref="electron",
                                     med="light",
                                     threshold=1e-3):
    """
    See documentation for rate.

    n_cl - number of events of a certain confidence level
    exp - total exposure of experiment

    Output - cross section constraint   units : cmet^2
    """

    rate = ph_dd_rate_numba(m_chi, rho_target, t,
                            q_grid, omega_grid, jac_q, jac_omega,
                            dyn_structure_factor,
                            ref=ref, med=med, rate=False, threshold=threshold)
    # problem
    exp = exposure * exposure_conversion
    # exp = to_unit_sys_mag(exposure, 'NateV')

    if rate <= 0.:
        return 0.
    else:
        # problem with to_unit_mag
        return (n_cl / (rate * exp)) * rate_conversion


@nb.jit
def mu_numba(m1, m2):
    return m1 * m2 / (m1 + m2)


@nb.jit
def med_form_factor_sq_numba(q_vec, m_chi, ref="electron", med="light"):
    """
    Mediator form factor

    Input units: q_vec - eV
                 m_chi - eV

    Output units: dimensionless
    """

    q_mag = np.linalg.norm(q_vec)

    if med == "light":
        power_q = 2
    elif med == "heavy":
        power_q = 0

    if ref == "electron":
        q0 = alpha_EM * m_elec
    elif ref == "nucleon":
        q0 = m_chi * v0
    else:
        print("error in mediator form factor")
        q0 = 0.

    return (q0 / q_mag) ** (2 * power_q)


@nb.jit
def kinematic_function_numba(q_vec, omega, t, m_chi):
    """
    g function = 2*pi*int d^3v f_chi delta(omega - omega_q)

    Input units: q_vec - eV
                 omega - eV
                 t - hr
                 m_chi - eV

    Output units - eV^(-1)
    """

    q_mag = np.linalg.norm(q_vec)

    v_minus = (q_mag) ** (-1) * np.abs(np.dot(q_vec, vE_vec_numba(t)) + 0.5 * q_mag ** 2 / m_chi + omega)

    if v_minus <= vEsc:
        return 2 * np.pi ** 2 * v0 ** 2 * (N0 * q_mag) ** (-1) * (
                    np.exp(-v_minus ** 2 / v0 ** 2) - np.exp(-vEsc ** 2 / v0 ** 2))
    else:
        return 0.


@nb.jit
def vE_vec_numba(t):
    """
    Earth velocity as a function of time

    t - hr
    """

    phi = 2 * np.pi * (t / 24)

    vE_x = vE * np.sin(thetaE) * np.sin(phi)
    vE_y = vE * np.sin(thetaE) * np.cos(thetaE) * (np.cos(phi) - 1)
    vE_z = vE * (np.cos(thetaE) ** 2 + np.sin(thetaE) ** 2 * np.cos(phi))

    return np.array([vE_x, vE_y, vE_z])
