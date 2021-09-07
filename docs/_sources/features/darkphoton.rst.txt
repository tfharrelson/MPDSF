*********************************************
Features: Dark photon mediated scattering
*********************************************

Dark photon mediated scattering assumes that the dark matter interacts with normal matter via a dark photon. This dark photon is kinetically mixed with normal photons, which means that there is a very low probability of observing it behave like a normal photon. Now, very low is not zero, so we can calculate its scattering response. Since it behaves like a normal photon, we simply apply the theory of photon-lattice interactions, by analyzing the Frohlich Hamiltonian that describes the interaction between photons and phonons. In the normal dynamic structure factor calculation, there are terms of the form:

:math:`\textbf{q} \cdot \boldsymbol{\epsilon}_{\tau, s\textbf{k}}`

which is the dot product between the momentum transfer vector, and the phonon eigenvector. There are other factors involved in the full calculation, but they are not important to the current discussion. The above expression essentialy assumes that the scattering particle is a billiard ball with no preferential interactions toward any atom in the lattice. In the case of photon scattering and absorption, the photons interact with the charges on the atoms. The Frohlich Hamiltonian models this interaction through the Born effective charges and dielectric tensor. The above expression changes to:

:math:`\frac{q^2}{\textbf{q} \cdot \overline{\boldsymbol{\epsilon}}_\infty \cdot \textbf{q}} \textbf{q} \cdot \overline{Z}^*_{\tau} \cdot \boldsymbol{\epsilon}_{\tau, s\textbf{k}}`

in which :math: `Z` is the Born effective charge tensor, and :math: `\boldsymbol{\epsilon}_\infty` is the high frequency dielectric tensor.

If the dark-photon tags are set in the input file, then this replacement is made and the calculation is run. Also, the code attempts to set the number of ``overtones`` to 1 because the second-order effects of this are not known, and it is not clear that the same convolution operation can be used. Thus, it is recommended to set the ``overtones`` tag to 1 in the input.
