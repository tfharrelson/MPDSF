import numpy as np
import re
import scipy.constants as const

class Dyn_System:
    def __init__(self,yaml_file):
        self.natoms = self.find_natoms(yaml_file)
        self.qpoints = self.import_qpts(yaml_file)
        self.positions = self.import_positions(yaml_file)
        self.lattice = self.import_lattice(yaml_file)
        self.masses = self.import_masses(yaml_file, self.natoms)
        self.frequencies = self.import_frequencies(yaml_file, 3 * self.natoms * len(self.qpoints))
        self.eigvecs = self.import_eigvecs(yaml_file, self.natoms, 3 * self.natoms * len(self.qpoints))
        self.supercell = self.import_supercell(yaml_file)
        self.rlattice = self.import_recip_lattice(yaml_file)
        self.weights = self.import_weights(yaml_file, len(self.qpoints))
        self.mesh = self.import_mesh(yaml_file)

    def find_natoms(self, yaml_file):
        ifileReader = open(yaml_file, 'r')
        check = 0
        for line in ifileReader:
            if check < 10:
                #print(line[0:5])
                check = check + 1
            if line[0:5] == "natom":
                #print(line[0:5])
                words = line.split()
                natoms = int(words[1])
                return natoms

    def import_positions(self, yaml_file):
        # consider changing this to account for the number of atoms to improve speed
        ifile_reader = open(yaml_file,'r')
        p = re.compile('[+-]?(\d+\.\d+)')
        #positions = np.array
        positions = np.empty([0,3],dtype=np.float)
        for line in ifile_reader:
            if line[2:10] == "position" or line[2:13] == "coordinates":
                pos = np.array([p.findall(line)]).astype(np.float)
                positions = np.append(positions, pos, 0)
        return positions

    def import_lattice(self, yaml_file):
        ifile_reader = open(yaml_file, 'r')
        lattice = np.zeros([3, 3], dtype=np.float)
        # lattice system always is a 3x3 matrix
        p = re.compile('[+-]?(\d+\.\d+)')
        vec = np.zeros(3)
        for line in ifile_reader:
            if line[0:8] == 'lattice:':
                counter = 1
                while counter <= 3:
                    line = next(ifile_reader)
                    words = p.findall(line)
                    for i in range(0, 3):
                        vec[i] = float(words[i])
                    #print(vec)
                    #norm_constant = np.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])
                    lattice[counter - 1, :] = vec #/ norm_constant
                    counter = counter + 1
                return lattice

    def import_masses(self, yaml_file, natoms):
        ifile_reader = open(yaml_file, 'r')
        mass_vec = np.zeros(natoms)
        i = 0
        for line in ifile_reader:
            if line[2:7] == 'mass:':
                words = line.split()
                mass_vec[i] = float(words[1])
                if i == natoms - 1:
                    return mass_vec
                else:
                    i = i + 1
    def import_frequencies(self, yaml_file, num_freqs):
        mesh_reader = open(yaml_file, 'r')

        p = re.compile('[+-]?(\d+\.\d+)')
        freq_list = np.zeros(num_freqs)
        i = 0
        for line in mesh_reader:
            if line[4:13] == "frequency":
                words = line.split()
                freq_list[i] = float(words[1])
                i = i + 1
        return freq_list

    def import_weights(self, yaml_file, num_weights):
        weights = np.zeros([num_weights])
        mesh_reader = open(yaml_file, 'r')

        counter = 0
        for line in mesh_reader:
            if line[2:8] == "weight":
                weights[counter] = float(line.split()[1])
                counter += 1

        return weights

    def import_eigvecs(self, yaml_file, natoms, num_eigs):
        meshReader = open(yaml_file, 'r')
        headerFlag = True
        eigvecs = np.zeros([natoms, 3, num_eigs], dtype=np.complex)
        p = re.compile('-?\d+\.\d+')
        eig_counter = 0
        num_qpts = int(num_eigs / (3 * natoms))
        for line in meshReader:
            eigvec = np.zeros([natoms, 3], dtype=np.complex)
            i = 0

            while line[4:13] != "frequency":
                #       while line[0:12]!="- q-position":
                #if line[2:8] == "weight":
                #    words = line.split()
                #    newWeight = float(words[1])
                if line[6:12] == "# atom":
                    line = next(meshReader)
                    #print('line is = '), print(line)
                    #nums = p.findall(line)
                    nums = np.array(p.findall(line)).astype(np.complex)
                    #print('nums ='), print(nums)
                    eigvec[i, 0] = nums[0] + 1j * nums[1]
                    #print('eigvec = '), print(eigvec)
                    line = next(meshReader)
                    nums = np.array(p.findall(line)).astype(np.complex)
                    eigvec[i, 1] = nums[0] + 1j * nums[1]
                    line = next(meshReader)
                    #nums = p.findall(line)
                    nums = np.array(p.findall(line)).astype(np.complex)
                    eigvec[i, 2] = nums[0] + 1j * nums[1]
                    i = i + 1
                    headerFlag = False
                try:
                    line = next(meshReader)
                except StopIteration:
                    eofFlag = 1
                    break
            # should have entire normalized mass-weighted eigenvector in eigVec now
            #print('Current eigenvector = ')
            #print(eigvec)
            if not headerFlag:
                #eigvecs = np.append(eigvecs, eigvec, 2)
                eigvecs[:, :, eig_counter] = eigvec
                eig_counter = eig_counter+1
        # return eigvecs
        # may potentially break other codes that rely on specific shape of eigvecs object
        # new shape is [num_atoms, cart_indices, num_qpts, num_branches]
        print('current eigvecs shape =', eigvecs.shape)
        print('example eigvec =', eigvecs[:,:,4])
        print('shape = [', natoms, 3, num_qpts, int(3*natoms),']')
        return np.reshape(eigvecs, [natoms, 3, num_qpts, int(3*natoms)])

    def import_supercell(self, yaml_file):
        ifile_reader = open(yaml_file,'r')
        p = re.compile('[+-]?(\d+)')
        supercell = np.zeros([3, 3], dtype=np.int)
        for line in ifile_reader:
            if line[0:16] == "supercell_matrix":
                # parse the next three lines to find the super cell
                for i in range(3):
                    vec = np.array([p.findall(next(ifile_reader))]).astype(np.int)
                    supercell[i, :] = vec
                return supercell
        return np.identity(3)

    def import_qpts(self, yaml_file):
        ifile_reader = open(yaml_file,'r')
        p = re.compile('[+-]?(\d+\.\d+)')
        qpoints = np.empty([0,3])
        for line in ifile_reader:
            if line[2:12] == "q-position":
                qpt = np.array([p.findall(line)]).astype(np.float)
                qpoints = np.append(qpoints, qpt, 0)
        return qpoints

    def import_recip_lattice(self, mesh_file):
        file_reader = open(mesh_file, 'r')
        rlattice = np.zeros([3, 3])
        # lattice system always is a 3x3 matrix
        p = re.compile('[+-]?(\d+\.\d+)')
        vec = np.zeros(3)
        for line in file_reader:
            if line[0:10] == 'reciprocal':
                counter = 1
                while counter <= 3:
                    line = next(file_reader)
                    words = p.findall(line)
                    for i in range(0, 3):
                        vec[i] = float(words[i])
                    print(vec)
                    #normConstant = np.sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2])
                    rlattice[counter - 1, :] = vec #/ normConstant
                    counter = counter + 1
                return 2 * np.pi * rlattice
    def import_mesh(self, mesh_file):
        p = re.compile('[+-]?(\d+)')
        file_reader = open(mesh_file, 'r')
        for line in file_reader:
            if line[0:4] == 'mesh':
                mesh = np.array(p.findall(line)).astype(int)
                return mesh

    def unnormalize_eigvecs(self):
        # conversion factor from THz (native units of pphonopy) to Hz
        conv_factor = 10**12
        mass_conv = np.sqrt(const.atomic_mass * self.masses[0])
        freq_conv = np.sqrt(const.hbar / (2 * conv_factor * self.frequencies[4]))
        num_bands = 3*self.natoms
        #num_qpts = len(self.qpoints)
        for i in range(len(self.masses)):
            self.eigvecs[i, :, :, :] = self.eigvecs[i, :, :, :] / np.sqrt(const.atomic_mass * self.masses[i])
        for i in range(len(self.frequencies)):
            if self.frequencies[i] < 0:
                #print('NEGATIVE FREQUENCY'), print(self.frequencies[i])
                #continue
                freq = 1e12
            else:
                freq = self.frequencies[i]
            q_index = int(np.floor(i / num_bands))
            band_index = i % num_bands
            self.eigvecs[:, :, q_index, band_index] = self.eigvecs[:, :, q_index, band_index] * np.sqrt(const.hbar / (2 * conv_factor * freq))
