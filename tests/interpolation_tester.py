import unittest
from src import Interpolation
from src.utils import Phono3pyInputs, PhononEigenvalues, PhononEigenvectors, ImaginarySelfEnergy, Phono3pyManager
import numpy as np

# For all tests, use the files in the data directory
poscar = '../data/GaAs/POSCAR'
fc2 = '../data/GaAs/fc2.hdf5'
fc3 = '../data/GaAs/FORCES_FC3'
disp = '../data/GaAs/disp_fc3.yaml'
mesh = [5, 5, 5]
supercell = np.diag([2, 2, 2])


class TestInterpolatorInstances(unittest.TestCase):
    def test_interpolate_eig_instance(self):
        inputs = Phono3pyInputs(poscar=poscar, fc3_file=fc3, disp_file=disp, mesh=mesh, supercell=supercell)
        eigs = PhononEigenvalues(inputs)
        interp = Interpolation.Interpolator(eigs)
        self.assertIsInstance(interp, Interpolation.Interpolator)

    def test_interpolate_eigvec_instance(self):
        inputs = Phono3pyInputs(poscar=poscar, fc3_file=fc3, disp_file=disp, mesh=mesh, supercell=supercell)
        eigvecs = PhononEigenvectors(inputs)
        interp = Interpolation.Interpolator(eigvecs)
        self.assertIsInstance(interp, Interpolation.Interpolator)

    def test_interpolate_ise_instance(self):
        inputs = Phono3pyInputs(poscar=poscar, fc3_file=fc3, disp_file=disp, mesh=mesh, supercell=supercell)
        ise = ImaginarySelfEnergy(inputs)
        interp = Interpolation.Interpolator(ise)
        self.assertIsInstance(interp, Interpolation.Interpolator)


class TestInterpolatorSetting(unittest.TestCase):
    def test_set_eigs(self):
        inputs = Phono3pyInputs(poscar=poscar, fc3_file=fc3, disp_file=disp, mesh=mesh, supercell=supercell)
        eigs = PhononEigenvalues(inputs)
        interp = Interpolation.Interpolator(eigs)
        interp.set_interpolators()
        self.assertIsInstance(interp, Interpolation.Interpolator)

    def test_set_eigvecs(self):
        inputs = Phono3pyInputs(poscar=poscar, fc3_file=fc3, disp_file=disp, mesh=mesh, supercell=supercell)
        eigvecs = PhononEigenvectors(inputs)
        interp = Interpolation.Interpolator(eigvecs)
        interp.set_interpolators()
        self.assertIsInstance(interp, Interpolation.Interpolator)

    def test_set_ise(self):
        inputs = Phono3pyInputs(poscar=poscar, fc3_file=fc3, disp_file=disp, mesh=mesh, supercell=supercell)
        ise = ImaginarySelfEnergy(inputs)
        interp = Interpolation.Interpolator(ise)
        interp.set_interpolators()
        self.assertIsInstance(interp, Interpolation.Interpolator)


class TestInterpolation(unittest.TestCase):
    def test_interp_eig(self):
        inputs = Phono3pyInputs(poscar=poscar, fc3_file=fc3, disp_file=disp, mesh=mesh, supercell=supercell)
        ise = PhononEigenvalues(inputs)
        interp = Interpolation.Interpolator(ise)
        interp.set_interpolators()
        test_q = interp.phase_space.qpoints[3]
        self.assertEqual(interp.interpolate(0, test_q)[0], interp.property.get_property_value(test_q, 0))

    def test_interp_eigvecs(self):
        inputs = Phono3pyInputs(poscar=poscar, fc3_file=fc3, disp_file=disp, mesh=mesh, supercell=supercell)
        ise = PhononEigenvectors(inputs)
        interp = Interpolation.Interpolator(ise)
        interp.set_interpolators()
        test_q = interp.phase_space.qpoints[3]
        self.assertEqual(interp.interpolate(0, test_q)[0], interp.property.get_property_value(test_q, 0)[0])

    def test_interp_ise(self):
        inputs = Phono3pyInputs(poscar=poscar, fc3_file=fc3, disp_file=disp, mesh=mesh, supercell=supercell)
        ise = ImaginarySelfEnergy(inputs)
        interp = Interpolation.Interpolator(ise)
        interp.set_interpolators()
        band_index = 4
        w_index = 3
        q = [0, 0, 0]

        # w index changes fastest for phase_space array
        test_ps = list(q) + [interp.property.freqs[w_index]]

        print('test phase space point =', test_ps)
        print(interp.interpolate(0, test_ps))
        self.assertFalse(interp._vector_flag)
        print(interp.property.freqs)
        self.assertEqual(np.array(interp.interpolate(0, *test_ps)), interp.property.get_property_value(q, band_index)[w_index])

if __name__ == '__main__':
    unittest.main()
