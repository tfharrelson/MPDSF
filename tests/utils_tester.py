import unittest
from src.utils import Phono3pyInputs, PhononEigenvalues, PhononEigenvectors, ImaginarySelfEnergy, Phono3pyManager
import numpy as np

# For all tests, use the files in the data directory
poscar = '../data/GaAs/POSCAR'
fc2 = '../data/GaAs/fc2.hdf5'
fc3 = '../data/GaAs/FORCES_FC3'
disp = '../data/GaAs/disp_fc3.yaml'
mesh = [5, 5, 5]
supercell = [2, 2, 2]

# instantiate manager to expedite tests
#inputs = Phono3pyInputs(poscar=poscar, fc3_file=fc3, disp_file=disp, mesh=mesh, supercell=supercell)
#manager = Phono3pyManager(inputs)

class TestInputs(unittest.TestCase):
    def test_inputs(self):
        print('testing Phono3py inputs...\n')
        test_inputs = Phono3pyInputs(poscar=poscar, fc3_file=fc3, disp_file=disp, mesh=mesh, supercell=supercell)
        self.assertIsInstance(test_inputs, Phono3pyInputs)
        self.assertEqual('../data/GaAs/POSCAR', test_inputs.poscar)
        self.assertListEqual([0, 0, 0], test_inputs.shift)
        print('ok\n')


class TestPhono3pyManager(unittest.TestCase):
    def test_instantiation(self):
        print('testing Phono3py manager instantiation...\n')
        test_inputs = Phono3pyInputs(poscar=poscar, fc3_file=fc3, disp_file=disp, mesh=mesh, supercell=supercell)
        test_manager = Phono3pyManager(test_inputs)
        self.assertIsInstance(test_manager, Phono3pyManager)
        print('ok\n')

    def test_eigvecs(self):
        print('testing Phono3py manager eigvec shape...\n')
        inputs = Phono3pyInputs(poscar=poscar, fc3_file=fc3, disp_file=disp, mesh=mesh, supercell=supercell)
        manager = Phono3pyManager(inputs)
        manager.set_phonons()
        self.assertEqual(len(manager.eigvecs.shape), 3)
        print('ok\n')

    def test_fix_phonon_data_module(self):
        print('testing the fix_phonon_data method in Phono3py manager')
        inputs = Phono3pyInputs(poscar=poscar, fc3_file=fc3, disp_file=disp, mesh=mesh, supercell=supercell)
        manager = Phono3pyManager(inputs)
        manager.phono3py.run_phonon_solver()
        eigs, vecs, qpoints = manager.phono3py.get_phonon_data()
        print('shape of eigs', eigs.shape)
        print('shape of vecs', vecs.shape)
        print('shape of qpoints', qpoints.shape)

        print('num qpoints =', len(eigs))
        print('expected num qpoints =', np.prod(manager.inputs.mesh))
        manager.set_phonons()
        print('new num of qpoints =', len(manager.bands))
        # first q-point should always be [0,0,0]
        self.assertListEqual(list(manager.qpoints[0]), [0., 0., 0.])
        print('first q-point =', manager.qpoints[0])


class TestEigenvalues(unittest.TestCase):
    def test_eig_shape(self):
        print('testing eigenvalue shape and type...\n')
        inputs = Phono3pyInputs(poscar=poscar, fc3_file=fc3, disp_file=disp, mesh=mesh, supercell=supercell)
        eigs = PhononEigenvalues(inputs)
        qpoint = np.array([0., 0., 0.])
        band = 0
        self.assertIsInstance(eigs.get_property_value(qpoint, band), np.float64)
        print('ok\n')
    def test_eig_value(self):
        print('testing that acoustic eigenvalues are close to zero...\n')
        inputs = Phono3pyInputs(poscar=poscar, fc3_file=fc3, disp_file=disp, mesh=mesh, supercell=supercell)
        eigs = PhononEigenvalues(inputs)
        qpoint = np.array([0., 0., 0.])
        band = 0
        thresh = 1e-2
        self.assertLess(np.abs(eigs.get_property_value(qpoint, band)), thresh)
        print('ok\n')


class TestEigenvectors(unittest.TestCase):
    def test_eigvec_instance(self):
        print('testing eigenvector instance...\n')
        inputs = Phono3pyInputs(poscar=poscar, fc3_file=fc3, disp_file=disp, mesh=mesh, supercell=supercell)
        eigs = PhononEigenvectors(inputs)
        qpoint = np.array([0., 0., 0.])
        band = 0
        self.assertIsInstance(eigs.get_property_value(qpoint, band), np.ndarray)
        print('ok\n')

    def test_eigvec_shape(self):
        print('testing eigenvector length...\n')
        inputs = Phono3pyInputs(poscar=poscar, fc3_file=fc3, disp_file=disp, mesh=mesh, supercell=supercell)
        eigs = PhononEigenvectors(inputs)
        qpoint = np.array([0., 0., 0.])
        band = 0
        self.assertEqual(len(eigs.get_property_value(qpoint, band)), 6)
        print('ok\n')

    def test_eigvec_values(self):
        print('testing eigenvector values..\n')
        inputs = Phono3pyInputs(poscar=poscar, fc3_file=fc3, disp_file=disp, mesh=mesh, supercell=supercell)
        eigs = PhononEigenvectors(inputs)
        qpoint = np.array([0., 0., 0.])
        band = 0
        # assert that eigvec in dict is same as eigvec in matrix
        dict_eigvec = eigs.get_property_value(qpoint, band)
        self.assertListEqual(list(eigs.manager.eigvecs[0, :, 0]), list(dict_eigvec))
        print('eigvec =', list(eigs.manager.eigvecs[0, :, 0]))
        print('ok\n')


class TestImagSelfEnergy(unittest.TestCase):
    def test_instantiation(self):
        inputs = Phono3pyInputs(poscar=poscar, fc3_file=fc3, disp_file=disp, mesh=mesh, supercell=supercell)
        ise = ImaginarySelfEnergy(inputs)
        qpoint = np.array([0., 0., 0.])
        band = 0
        self.assertIsInstance(ise.get_property_value(qpoint, band), np.ndarray, msg='ISE is a ndarray as intended!')

    def test_shape(self):
        inputs = Phono3pyInputs(poscar=poscar, fc3_file=fc3, disp_file=disp, mesh=mesh, supercell=supercell)
        ise = ImaginarySelfEnergy(inputs)
        qpoint = np.array([0., 0., 0.])
        #band = 0
        self.assertEqual(len(ise.get_imag_self_energies_at_q(qpoint).shape), 2, msg='ISEs are the proper shape!')


if __name__ == '__main__':
    unittest.main()
