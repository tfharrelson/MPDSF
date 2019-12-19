import numpy as np
import yaml_phonons
import h5py

class Anh_System:
    def __init__(self, yaml_file, hdf5_file):
        self.dyn_system = yaml_phonons.Dyn_System(yaml_file)
        self.gammas = self.import_gammas(hdf5_file)
    def import_gammas(self, hdf5_file):
        f = h5py.File(hdf5_file, 'r')
        # Assume only one temperature needed?
        temp_gammas = np.array(f['gamma'])[0]   # [0] for getting the gammas at the first temperature in the set
        gammas = np.zeros(np.prod(temp_gammas.shape))
        index = 0
        for i in range(temp_gammas.shape[0]):
            for j in range(temp_gammas.shape[1]):
                gammas[index] = temp_gammas[i, j]
                index += 1
        return gammas
