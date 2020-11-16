from src.utils import PhononEigenvalues, Phono3pyInputs
from src.Interpolation import Interpolator
import numpy as np
import matplotlib.pyplot as plt

poscar = '../data/GaAs/POSCAR'
fc2 = '../data/GaAs/fc2.hdf5'
fc3 = '../data/GaAs/FORCES_FC3'
disp = '../data/GaAs/disp_fc3.yaml'
mesh = [5, 5, 5]
supercell = np.diag([2, 2, 2])

if __name__ == '__main__':
    inputs = Phono3pyInputs(poscar=poscar, fc3_file=fc3, disp_file=disp, mesh=mesh, supercell=supercell)
    eigs = PhononEigenvalues(inputs)
    interp = Interpolator(eigs)

    # create plot for native eigenvalue datapoints
    ax = plt.gca()
    # mesh is [5,5,5] by default, so take the first three values
    qvals = []
    wvals = []
    for i in range(3):
        q = eigs._brillouinzone.qpoints[i]
        w = eigs.get_property_value(q, band_index=0)
        qvals.append(q[0])
        wvals.append(w)
    ax.scatter(qvals, wvals)

    new_qvals = [[q, 0, 0] for q in np.linspace(0, 0.5, 100)]
    new_wvals = np.array(interp.interpolate(0, new_qvals))

    ax.plot(np.array(new_qvals)[:, 0], new_wvals, color='k')
    plt.show()