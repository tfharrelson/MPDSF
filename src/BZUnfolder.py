import spglib as spg
import numpy as np

class BZUnfolder(object):
    def __int__(self,
                irr_property,
                irr_qpoints,
                map=None,
                cell=None,
                mesh=None,
                is_shift=[0,0,0]):
        """
        BZUnfolder is an object that unfolds property values defined in the irreducible BZ onto the full BZ
        :param property: property defined on the irreducible qpoint space; shape needs to be [num_qpts, ...]
        :param irr_qpoints: list of sampled qpoints in the irreducible BZ to be unfolded
        :param map: an optional mapping obtained from some phonopy outputs, which provides the mapping of the full set of qpoints to the irreducible
        :param cell: a tuple of (lattice vectors, atom positions, atom types)
        :param mesh: a list of integers describing the MP mesh grid
        :param is_shift: a vector describing the shift from gamma during mesh creation
        :return: Nothing
        """
        self.irr_property = irr_property
        self.irr_qpoints = irr_qpoints
        self.qpoints = None
        self.property = None
        self.map = None

    def unfold(self):
        if self.map is None:
            self.set_map()
        else:
            self.set_qpoints()
        self.unfold_map()

    def set_map(self):
        self.map, gridpoints = spg.get_ir_reciprocal_mesh(self.mesh, self.cell, self.is_shift)
        self.qpoints = np.divide(gridpoints, self.mesh)

    def set_qpoints(self):
        self.qpoints = np.zeros([np.prod(self.mesh), 3])
        count = 0
        curr_qx = 0.0
        curr_qy = 0.0
        curr_qz = 0.0
        spacing = 1.0 / np.array(self.mesh)
        for z in range(self.mesh[2]):
            for y in range(self.mesh[1]):
                for x in range(self.mesh[0]):
                    self.qpoints[count, :] = [curr_qx, curr_qy, curr_qz]
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

    def unfold_map(self):
        self.property = np.empty(len(self.qpoints))
        for i, id in enumerate(self.map):
            self.property[i] = self.irr_property[id]
