****************************************
Features: Scalar mediator scattering
****************************************

In the model of scalar mediated dark matter scattering, the scattering interaction is assumed to be proportional to the atomic mass, which means that the contributions from the dynamic structure factor to the differential scattering rate are weighted toward contributions to heavier atoms. Essentially, to calculate the scattering response of dark matter, the internal weights contributing to the full dynamic structure factor need to be changed. For scalar mediated dark matter, this turns out to be relatively simple, because the dynamic structure factor takes this form:

:math:`S(\textbf{q}, \omega) \propto \sum_{\tau_1, \tau_2} M_{\tau_1}^* M_{\tau_2}^*`

where :math:`M` is a matrix element connecting the ground state to a final state through the scattering interaction. The details are in the Theory sections of this site. The point here is that :math:`S` is calculated as a sum over atomic indices. To re-weight this for the scalar mediator model, we simply change the above equation to:

:math:`S(\textbf{q}, \omega) \propto \sum_{\tau_1, \tau_2} A_{\tau_1} M_{\tau_1}^* A_{\tau_2} M_{\tau_2}^*`

in which :math:`A` is the atomic mass of the atom. All other objects remain exactly the same as they do during normal dynamic structure factor calculation.

