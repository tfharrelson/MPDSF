*********************************************
Features: Including low-q scaling behavior
*********************************************

MPDSF has the capability to sample the low-q behavior via linear interpolation of phonon eigenvalues, phonon eigenvectors, and imaginary self energies. In principle, any low-q point in the dynamic structure factor can be calculated in this way, but currently it is not completely set up. Right now, the code samples q-points within the central voxel of the Monkhorst-Pack grid closest to the Gamma point, by discretizing that voxel into another Monkhorst-Pack grid. In two dimensions, I can draw an example below

3x3 MP grid in a two-dimensional space
#######################################


+-----+-----+-----+
|     |     |     |
|     |     |     |
|     |     |     |
+-----+-----+-----+
|     |     |     |
|     |Gamma|     |
|     |     |     |
+-----+-----+-----+
|     |     |     |
|     |     |     |
|     |     |     |
+-----+-----+-----+

3x3 MP grid with low-q sampling
###############################


+-----+-----+-----+
|     |     |     |
|     |     |     |
|     |     |     |
+-----+-+-+-+-----+
|     | | | |     |
|     +-+-+-+     |
|     | |G| |     |
|     +-+-+-+     |
|     | | | |     |
+-----+-+-+-+-----+
|     |     |     |
|     |     |     |
|     |     |     |
+-----+-----+-----+

Here, the lines between the cells in the table separate the sampled points in the Brillouin zone. In other words, the sample points are in the middle of each cell. Here, we see that the low-q sampling is simply another 3x3 grid inside of the middle area closest to the Gamma point

Here, the lines between the cells in the table separate the sampled points in the Brillouin zone. In other words, the sample points are in the middle of each cell. Here, we see that the low-q sampling is simply another 3x3 grid inside of the middle area closest to the Gamma point. 

In this scheme, the low-q sampled points are not on a regular grid anymore with respect to the full Brillouin zone, and thus the contact interactions cannot be calculated because we need Fourier transforms to calculate the contact interactions, and Fourier transforms can only be performed on a regular grid. A regular grid is defined as a discrete set of points that are equally spaced in each direction. The spacing in the (e.g.) x and y directions do not need to be the same. 
