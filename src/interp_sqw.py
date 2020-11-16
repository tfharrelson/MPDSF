import h5py
import numpy as np
from scipy.interpolate import griddata, LinearNDInterpolator
import sys
import pickle

filename = sys.argv[1]

with h5py.File(filename, 'r') as f1:

    s1 = np.array([sqw_at_q for sqw_at_q in f1['sqw']])
    qpts = np.array(f1['q-points'])
    reclat = np.array(f1['reclat'])
    freqs = np.array(f1['frequencies'])
    points1 = [list(q) + [t] for q in qpts for t in freqs]
    print(s1.shape)
    print(np.array(points1).shape)
    grid_x1, grid_y1, grid_z1, grid_w1 = np.mgrid[min(qpts[:, 0]):max(qpts[:, 0]):15j,
                                         min(qpts[:, 1]):max(qpts[:, 1]):15j,
                                         min(qpts[:, 2]):max(qpts[:, 2]):15j,
                                         min(freqs):max(freqs):len(freqs ) *1j]
    # grid_x2, grid_y2 = np.mgrid[0:q_x_2[-1]:1000j, min(freqs_2):max(freqs_2):len(freqs_2)*1j]
    # interp_grid = griddata(points1, s1.reshape(-1), (grid_x1, grid_y1, grid_z1, grid_w1),
    #                        method = 'linear', fill_value = 0., rescale = True)
    interp_LND1 = LinearNDInterpolator(points1, s1.reshape(-1), fill_value=0.)

    if len(sys.argv) == 2:
        outfilename = 'interpolator'
    else:
        outfilename = sys.argv[2]

    outfile = open(outfilename, 'wb')
    pickle.dump(interp_LND1)