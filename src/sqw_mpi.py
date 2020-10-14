#!/usr/bin/env python
from src import yaml_phonons as ph, compute_Sqw
from mpi4py import MPI
import sys
import numpy as np
import h5py

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

yaml_filename = sys.argv[1]
hdf5_filename = sys.argv[2]
phonons = ph.Dyn_System(yaml_filename)
qpts = phonons.qpoints

coh_flag = True

#qpt_list = qpts[rank::size, :]

#qpt_list = comm.Scatterv(qpts, root=0)
qpt_sizes = np.zeros(size,dtype=int)
count = 0
for i in range(len(qpts[:,0])):
    qpt_sizes[count] += 1
    count += 1
    if count == size:
        count = 0
if rank == 0:
    print('qpt_sizes =', 3 * qpt_sizes)
#qpt_sizes *= 3
qpt_list = np.zeros([qpt_sizes[rank], 3])
qpt_sizes *= 3
print('qpt_list.shape =', qpt_list.shape)
#if rank==0:
displ_sizes = np.insert(np.cumsum(qpt_sizes), 0, 0)[0:-1]
print('displ-sizes =', displ_sizes)
sendbuf = [qpts, qpt_sizes, np.insert(np.cumsum(qpt_sizes),0,0)[0:-1], MPI.DOUBLE]
#else:
#    sendbuf = None
#sendbuf = comm.bcast(sendbuf, root=0)
comm.Scatterv(sendbuf, qpt_list, root=0)
print('rank =', rank)
print('qpt list =', qpt_list)
if rank == 0:
    print('full qpt list =', qpts)
#for q in qpt_list:
#    s_qw = compute_Sqw.Runner.get_coherent_spectrum(yaml_filename, hdf5_filename, q, num_overtones=10, delta_e=0.01, max_e=30)
#partial_sqw_list = np.array([compute_Sqw.Runner(yaml_filename,hdf5_filename, q_point=q,
#                                                 num_overtones=1, delta_e=0.01, max_e=30).get_coherent_spectrum()
#                             for q in qpt_list])
dsf = compute_Sqw.DynamicStructureFactor(poscar, fc, mesh, supercell, qpt_list, delta_e=de, max_e=me, num_overtones=no)
dsf.get_coherent_sqw()
#dsf.write_coherent_sqw()
if rank == 0:
    #test_qpts = np.zeros([int(sum(qpt_sizes)/3), 3])
    #rec_sizes = qpt_sizes
    #rec_disp = displ_sizes
    partial_sqw_shape = partial_sqw_list.shape
    full_sqw_shape = np.array(partial_sqw_shape)
    full_sqw_shape[0] = len(qpts)
    full_sqw_list = np.zeros(full_sqw_shape)
    #comm.bcast(full_sqw_shape, root=0)
    rec_sizes = qpt_sizes / 3 * np.prod(full_sqw_shape[1:])
    rec_sizes = rec_sizes.astype(int)
    rec_disp = np.insert(np.cumsum(rec_sizes), 0, 0)[0:-1]
else:
    #test_qpts = None
    rec_sizes = None
    rec_disp = None
    full_sqw_list = None
full_sqw_list = comm.bcast(full_sqw_list, root=0)
rec_sizes = comm.bcast(rec_sizes, root=0)
rec_disp = comm.bcast(rec_disp, root=0)
comm.Barrier()
#comm.Gatherv(qpt_list, [test_qpts, rec_sizes, rec_disp, MPI.DOUBLE], root=0)
comm.Gatherv(np.abs(partial_sqw_list), [full_sqw_list, rec_sizes, rec_disp, MPI.DOUBLE], root=0)
#comm.Gatherv(partial_sqw_list, [full_sqw_list, rec_sizes], root=0)
delta_e = 0.01
if rank==0:
    #print('gathered qpt_list =', test_qpts)
    for i, q in enumerate(qpts):
        (j, k, l) = np.round(q * phonons.mesh).astype(int)
        print('indices used =', j,k,l)
        print('integral for q =', q, 'is =', np.trapz(full_sqw_list[i, j, k, l, :]) * delta_e)
    hf = h5py.File('sqw_output.hdf5', 'w')
    hf.create_dataset('s_qw', data=full_sqw_list)
    hf.create_dataset('qpts', data=qpts)
    hf.close()
#full_sqw_list = comm.Gatherv(partial_sqw_list, root=0)
