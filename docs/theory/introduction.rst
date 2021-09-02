********************
Theory: Introduction
********************

The goal of computing a dynamic structure factor is to evaluate the following function:

:math:`S(\textbf{q}, \omega) = \frac{1}{N} \sum_f \left| \sum_{l}^N \sum_\tau^n \langle \Phi_f | e^{i\textbf{q}\cdot \textbf{r}_{l\tau}} | 0\rangle \right|^2 \delta(E_f - \omega)`

in which :math:`N` is the number of supercells (indexed by :math:`l`), :math:`\tau` is the atomic indices, :math:`\Phi_f` is the final state wavefunction after the scattering event, :math:`\textbf{q}` is the momentum of the scattering event, :math:`\textbf{r}_{l\tau}` is the position operator of atom :math:`\tau` in supercell :math:`l`, :math:`E_f` is the final state energy, and :math:`\omega` is the frequency/energy transfer from the scatttering event. This equation implicitly assumes that the initial state is in the vacuum ground state with no phonons present, so the initial energy is set to 0 and ignored. The final state is assumed to be some multi-phonon state comprised of the phonons calculated from the harmonic approximation in Phono3py. This multi-phonon state can be visualized as a vector/string of occupation numbers, in which the index of the vector corresponds to a specific :math:`s,\textbf{k}` combination; e.g. :math:`|\Phi_f\rangle = |001000\rangle` for the case in which we are only considered one :math:`\textbf{k}` value (usually the Gamma point), and there are 6 phonon branches/:math:`s` indices, which is true for a diatomic unit cell. In that example, there is one phonon in the third phonon branch, and zero phonons elsewhere in the manifold. To compute the matrix elements of the above operator, expand the position operator into:

:math:`\textbf{r}_{l\tau}= \textbf{R}_l + \textbf{r}_\tau^\circ + \textbf{u}_\tau`

which splits the contributions to the position operator into the lattice vector position, the average position of the atom in the unit cell, and the deviation of the atomic position due to the motion of the atoms. The :math:`u` term is the only dynamical variable here, the others are constant vectors which do not change the state function. In the harmonic limit, one can express the :math:`\textbf{u}` operator in terms of raising and lowering operators via:

:math:`\textbf{u}_{l\tau} = \frac{1}{\sqrt{N}} \sum_{s,\textbf{k}} \sqrt{\frac{\hbar}{2m_\tau\omega}}(\hat{a}_{s\textbf{k}} + \hat{a}_{s-\textbf{k}}^{\dagger}) \boldsymbol{\epsilon}_{\tau,s\textbf{k}} e^{i\textbf{k}\cdot\textbf{R}_l}`

where the raising and lowering operators (called ladder operators) are denoted by :math:`\hat{a}`, and :math:`\boldsymbol{\epsilon}` is the phonon mode eigenvector. With this expression the matrix elements can be written down, although the process is tedious, especially because the displacement operator is in the exponent of an exponential function. Therefore, we are applying a power series of these ladder operators, and keeping track of the indices in a code is not the easiest task. Fortunately, we can express these matrix elements with a series of convolution operations which account for all the index tracking automatically. This means that computationally we need only define a convolution operation and the objects to convolve. 

We also implemented the possibility of anharmonic broadening to the dynamic structure factor. The equation at the top of the page is still correct because we generically defined :math:`E_f` to be whatever we need it to be. However, since we are expressing the final states in terms of the harmonic phonons, :math:`E_f` must take the form of a sum over the phonon energies that are in the multi-phonon state. For the 1-phonon final states that means :math:`E_f` must be equal to a phonon energy/eigenvalue, meaning that the dynamic structure factor has delta functions at these energies. The anharmonic broadening is generally considered to be a weak perturbation, which allows us to approximate these delta functions as Lorentzian type functions of the form:

:math:`\delta(\omega - \omega_{sk}) \rightarrow \frac{1}{\pi} \frac{\Gamma_{sk}(\omega)}{(\omega - \omega_{sk})^2 + [\Gamma_{sk}(\omega)]^2}`

in which :math:`\Gamma(\omega)` is the imaginary part of the self-energy, which is a function of :math:`\omega`. In the limiting case, that :math:`\Gamma` is a constant, the above equation becomes exactly a Lorentzian distribution. However, when :math:`\omega >> \omega_{sk}`, then the above distribution function no longer resembles a Lorentzian.

When the anharmonic and contact interactions are included in the calculation of :math:`S(\textbf{q},\omega)`, then we obtain the multi-phonon dynamic structure factor, which is whole point of this code. 

