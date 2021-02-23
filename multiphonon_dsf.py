import argparse
import numpy as np
from src.compute_Sqw import DynamicStructureFactor
import yaml
from yaml import SafeLoader

class MPDSF:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-p", "--poscar", help='VASP poscar file of unit cell.', default='POSCAR')
        parser.add_argument("-fc3", "--fc3", help='Third order force constants file in hdf5 format.',
                            default=None)
        parser.add_argument('-fc2', '--fc2', help='Second order force constants file in FORCE_SETS format.',
                            default='FORCE_SETS')
        parser.add_argument('-d', '--disp', help='Disp.yaml file that comes with phono3py fc3 calculations.',
                            default='phonopy_disp.yaml')
        parser.add_argument('-m', '--mesh',
                            help='List of mesh values specifying the Monkhorst-Pack mesh for the phonon '
                                 'calculations. List is delimited by commas, no spaces. Ex// -m 3,3,3',
                            default='5,5,5')
        parser.add_argument('-s', '--supercell', help='List of supercell indices in the same format as the mesh input',
                            default='2,2,2')
        parser.add_argument('-e', '--max_energy', type=float, help='Maximum energy of spectrum in THz',
                            default=100.)
        parser.add_argument('--delta_e', type=float, help='Size of the energy bins used in the calculation in THz',
                            default=1.)
        parser.add_argument('-o', '--overtones', type=int,
                            help='Number of overtones to include in the multiphonon calculation',
                            default=2)
        parser.add_argument('--output',
                            help='Name of the output file that stores the dynamic structure factor information '
                                 'in hdf5 format. Default is sqw_output.hdf5',
                            default='sqw_output.hdf5')
        parser.add_argument('--numG', type=int, help='Number of reciprocal lattice vectors to include in q-point list.',
                            default=1)
        parser.add_argument('--strideQ', type=int,
                            help='Number of q-points to skip on the MP grid; still in testing so '
                                 'prepare for error messages :/... This really only works with certain'
                                 'MP grid, stride pairs; for instance a stride of 4 does not work with'
                                 'a 7,7,7 grid because 4 does not divide evenly into 7',
                            default=1)
        parser.add_argument('--qmax', type=float, help='(Optional) Set maximum q allowed in sampled q-points. Useful'
                                                       'when contructing a q-point list from scratch.',
                            default=None)
        parser.add_argument('--shift', help='List of shifts to apply to MP grid.',
                            default='0,0,0')
        parser.add_argument('-i', '--input',
                            help='Name of input file containing all these flags. Contents in this file will '
                                 'take precedence over command line arguments.')
        parser.add_argument('--mpi', help='Flag specifying if MPI ranks are to be used', action='store_true')
        parser.add_argument('-b', '--born', help='BORN file containing born effective charges and dielectric tensor.'
                                                 'This file is computed using VASP tag LEPSILON=.TRUE. and converted'
                                                 'to BORN format using the phonopy-vasp-born command.',
                            default=None)
        parser.add_argument('--scalar_mediator', help='Flag specifying if S(q,w) to be calculated for scalar mediator'
                                                      'calculations', action='store_true')
        parser.add_argument('--dark_photon', help='Flag specifying if S(q,w) to be calculated for dark-photon '
                                                  'calculations', action='store_true')


        args = parser.parse_args()
        self.poscar = args.poscar
        self.fc2 = args.fc2
        self.fc3 = args.fc3
        self.disp = args.disp
        self.mesh = np.array(args.mesh.split(',')).astype(np.int)
        self.supercell = np.array(args.supercell.split(',')).astype(np.int)
        self.max_e = args.max_energy
        self.delta_e = args.delta_e
        self.overtones = args.overtones
        self.output = args.output
        self.numG = args.numG
        self.strideQ = args.strideQ
        self.shift = np.array(args.shift.split(',')).astype(np.float)
        #TODO decide whether implementing a stride in G is worth it
        self.strideG = 1
        self.qmax = args.qmax
        self.qpoints = None
        #TODO Implement code that transforms qpoints into irreducible qpoints depending on the symmetry of the lattice
        self.weights = None

        self.input = args.input
        self.mpi = args.mpi
        self.dsf = None
        self.born = args.born
        if self.born is not None:
            self.is_nac = True
        else:
            self.is_nac = False
        self.dark_photon_flag = args.dark_photon
        if self.dark_photon_flag:
            self.overtones = 1
            if self.fc3 is None:
                print('I HOPE YOU KNOW WHAT YOU\'RE DOING... THERE IS NO FC3 FILE SET AND YOU WANT TO RUN THE DARK'
                      'PHOTON VARIANT OF THE CODE. YOU\'D LIKELY BE BETTER OFF RUNNING A DIFFERENT CODE IN THIS CASE...'
                      'For example, see https://github.com/tanner-trickle/dm-phonon-scatter')
        self.scalar_mediator_flag = args.scalar_mediator

        if self.input is not None:
            self.parse_input_file(self.input)
        self.set_qpoints()
        print('final qpoints are', self.qpoints)

    def parse_input_file(self, input_file):
        print('input file is', input_file)
        config_dict = yaml.load(open(input_file, 'rb'), Loader=SafeLoader)
        print('after loading, dict is ', config_dict)
        if 'poscar' in config_dict.keys():
            self.poscar = config_dict['poscar']
        if 'fc2' in config_dict.keys():
            self.fc2 = config_dict['fc2']
        if 'fc3' in config_dict.keys():
            self.fc3 = config_dict['fc3']
        if 'disp' in config_dict.keys():
            self.disp = config_dict['disp']
        if 'mesh' in config_dict.keys():
            self.mesh =config_dict['mesh']
        if 'supercell' in config_dict.keys():
            self.supercell = config_dict['supercell']
        if 'max_energy' in config_dict.keys():
            self.max_e = config_dict['max_energy']
        if 'delta_e' in config_dict.keys():
            self.delta_e = config_dict['delta_e']
        if 'overtones' in config_dict.keys():
            self.overtones = config_dict['overtones']
        if 'output' in config_dict.keys():
            self.output = config_dict['output']
        if 'numG' in config_dict.keys():
            self.numG = config_dict['numG']
        if 'strideQ' in config_dict.keys():
            self.strideQ = config_dict['strideQ']
        if 'shift' in config_dict.keys():
            self.shift = config_dict['shift']
        # TODO decide whether implementing a stride in G is worth it
        if 'strideG' in config_dict.keys():
            self.strideG = 1
        if 'qmax' in config_dict.keys():
            self.qmax = config_dict['qmax']
        self.qpoints = None
        # TODO Implement code that transforms qpoints into irreducible qpoints depending on the symmetry of the lattice
        self.weights = None

        self.set_qpoints()

        if 'mpi' in config_dict.keys():
            self.mpi = config_dict['mpi']
        self.dsf = None
        if 'born' in config_dict.keys():
            self.born = config_dict['born']
        if self.born is not None:
            self.is_nac = True
        else:
            self.is_nac = False
        if 'dark_photon' in config_dict.keys():
            self.dark_photon_flag = config_dict['dark_photon']
        if self.dark_photon_flag:
            self.overtones = 1
            if self.fc3 is None:
                print('I HOPE YOU KNOW WHAT YOU\'RE DOING... THERE IS NO FC3 FILE SET AND YOU WANT TO RUN THE DARK'
                      'PHOTON VARIANT OF THE CODE. YOU\'D LIKELY BE BETTER OFF RUNNING A DIFFERENT CODE IN THIS CASE...'
                      'For example, see https://github.com/tanner-trickle/dm-phonon-scatter')
        if 'scalar_mediator' in config_dict.keys():
            self.scalar_mediator_flag = config_dict['scalar_mediator']

    def set_qpoints(self):
        self.qpoints = np.zeros([0, 3])
        count = 0
        curr_qx = self.shift[0] / self.mesh[0]
        curr_qy = self.shift[1] / self.mesh[1]
        curr_qz = self.shift[2] / self.mesh[2]
        spacing = float(self.strideG) / np.array(self.mesh)
        for gz in range(0, self.numG, self.strideG):
            for gy in range(0, self.numG, self.strideG):
                for gx in range(0, self.numG, self.strideG):
                    for z in range(0, self.mesh[2], self.strideQ):
                        for y in range(0, self.mesh[1], self.strideQ):
                            for x in range(0, self.mesh[0], self.strideQ):
                                if self.qmax is not None:
                                    if np.abs(curr_qx) <= self.qmax and np.abs(curr_qy) <= self.qmax and np.abs(
                                            curr_qz) <= self.qmax:
                                        self.qpoints = np.append(self.qpoints,
                                                                 [[curr_qx + gx, curr_qy + gy, curr_qz + gz]],
                                                                 axis=0)
                                else:
                                    self.qpoints = np.append(self.qpoints,
                                                             [[curr_qx + gx, curr_qy + gy, curr_qz + gz]],
                                                             axis=0)
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
        # Remove the first point, which is at gamma (q = [0, 0, 0]) because it doesn't do anything
        self.qpoints = self.qpoints[1:]
        return self.qpoints

    def run(self, start_q_index=None, stop_q_index=None):
        self.dsf = DynamicStructureFactor(self.poscar,
                                          self.fc2,
                                          self.mesh,
                                          self.supercell,
                                          self.qpoints,
                                          self.shift,
                                          self.fc3,
                                          disp_file=self.disp,
                                          delta_e=self.delta_e,
                                          max_e=self.max_e,
                                          num_overtones=self.overtones,
                                          is_nac=self.is_nac,
                                          born_file=self.born,
                                          scalar_mediator_flag=self.scalar_mediator_flag,
                                          dark_photon_flag=self.dark_photon_flag)
        self.dsf.get_coherent_sqw(start_q_index=start_q_index, stop_q_index=stop_q_index)

    def save_data(self):
        if self.dsf is None:
            print('ERROR: Need to execute the run command to create the multiphonon dynamic structure factor object.')
        else:
            self.dsf.write_coherent_sqw(self.output)

if __name__ == '__main__':

    mpdsf = MPDSF()
    if mpdsf.mpi is True:
        # Import mpi4py
        from mpi4py import MPI

        # split up q-points into partial lists and run separately on different MPI processes
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        num_qpts_per_rank = np.ceil(len(mpdsf.qpoints) / size).astype(int)
        start_q_index = int(rank * num_qpts_per_rank)
        stop_q_index = int((rank + 1) * num_qpts_per_rank)
        if stop_q_index > len(mpdsf.qpoints):
            stop_q_index = len(mpdsf.qpoints)
        mpdsf.run(start_q_index=start_q_index, stop_q_index=stop_q_index)

        # Place barrier to make sure all MPI processes wait for each other
        comm.Barrier()

        full_sqw_list = np.zeros([len(mpdsf.qpoints), len(mpdsf.sqw[0])])
        # Find rec sizes, in which the last rank may have a different number of q-points
        rec_sizes = np.array(int(size) * [num_qpts_per_rank])
        rec_sizes[-1] = len(mpdsf.qpoints) % num_qpts_per_rank
        rec_sizes *= len(mpdsf.sqw[0])

        # Find the rec displacements, and we don't want to skip any data, so a cumsum will work
        rec_disp = np.insert(np.cumsum(rec_sizes), 0, 0)[0:-1]

        # Gather all objects into rank 0 in full_sqw_list
        comm.Gatherv(np.abs(mpdsf.sqw), [full_sqw_list, rec_sizes, rec_disp, MPI.DOUBLE], root=0)
        if rank == 0:
            mpdsf.sqw = full_sqw_list
            mpdsf.save_data()
    else:
        mpdsf.run()
        mpdsf.save_data()
    # Import q-point list somehow...?
    # Options include specifying a number of G-points to go out to, and calculate for the entire mesh
    # Another option is to read a simple text file with each q-point on each line
    # Probably both is a good option

    #Also a good idea may be to include an option for an input file, where all these flags are contained
    # to reduce the command line clutter