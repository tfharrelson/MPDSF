import sympy
from sympy import symbols
from sympy import N

hbar = 1.054571817*10**(-34) # J*s
c_light = 2.99792458*10**8 # m/s
k_boltz = 1.380649*10**(-23) # J/K
m_sol = 1.98847*10**(30) # kg
parsec = 3.086*10**(16) # m

J_to_eV = 1/(1.60217662*10**(-19))

# distance
umet, mmet, cmet, met, kmet = symbols('umet mmet cmet met kmet')
Ang = symbols('Ang')
pc, kpc, Mpc, Gpc = symbols('pc kpc Mpc Gpc')

# energy
ueV, meV, eV, keV, MeV, GeV, TeV, PeV = symbols('ueV meV eV keV MeV GeV TeV PeV')
joule = symbols('joule')

# frequency
mHz, Hz, kHz, MHz, GHz, THz = symbols('mHz Hz kHz MHz GHz THz')

# temperature
mkel, kel = symbols('mkel kel')

# time
usec, msec, sec = symbols('usec msec sec')
minute, hour, day, week, year = symbols('minute hour day week year')

# mass
grm, kgrm = symbols('grm kgrm')
msol = symbols('msol')

# dictionary which converts all of the values
# dict_key = dict_val

# to define a new unit system simply add a new dictionary with all conversions
convert_dict = {
    'NateV': {
        # distance
        umet: 10 ** (-6) * (hbar * c_light) ** (-1) * (J_to_eV) ** (-1) * eV ** (-1),
        mmet: 10 ** (-3) * (hbar * c_light) ** (-1) * (J_to_eV) ** (-1) * eV ** (-1),
        cmet: 10 ** (-2) * (hbar * c_light) ** (-1) * (J_to_eV) ** (-1) * eV ** (-1),
        met: (hbar * c_light) ** (-1) * (J_to_eV) ** (-1) * eV ** (-1),
        kmet: 10 ** (3) * (hbar * c_light) ** (-1) * (J_to_eV) ** (-1) * eV ** (-1),
        Ang: 10 ** (-10) * (hbar * c_light) ** (-1) * (J_to_eV) ** (-1) * eV ** (-1),

        pc: parsec * (hbar * c_light) ** (-1) * (J_to_eV) ** (-1) * eV ** (-1),
        kpc: 10 ** 3 * parsec * (hbar * c_light) ** (-1) * (J_to_eV) ** (-1) * eV ** (-1),
        Mpc: 10 ** 6 * parsec * (hbar * c_light) ** (-1) * (J_to_eV) ** (-1) * eV ** (-1),
        Gpc: 10 ** 9 * parsec * (hbar * c_light) ** (-1) * (J_to_eV) ** (-1) * eV ** (-1),

        # energy
        ueV: 10 ** (-6) * eV,
        meV: 10 ** (-3) * eV,
        keV: 10 ** 3 * eV,
        MeV: 10 ** 6 * eV,
        GeV: 10 ** 9 * eV,
        TeV: 10 ** 12 * eV,
        PeV: 10 ** 15 * eV,

        joule: J_to_eV * eV,

        # frequency
        mHz: 10 ** (-3) * hbar * J_to_eV * eV,
        Hz: hbar * J_to_eV * eV,
        kHz: 10 ** 3 * hbar * J_to_eV * eV,
        MHz: 10 ** 6 * hbar * J_to_eV * eV,
        GHz: 10 ** 9 * hbar * J_to_eV * eV,
        THz: 10 ** 12 * hbar * J_to_eV * eV,

        # temperature
        mkel: 10 ** (-3) * k_boltz * J_to_eV * eV,
        kel: k_boltz * J_to_eV * eV,

        # time
        usec: 10 ** (-6) * (hbar) ** (-1) * (J_to_eV) ** (-1) * eV ** (-1),
        sec: (hbar) ** (-1) * (J_to_eV) ** (-1) * eV ** (-1),

        minute: 60 * (hbar) ** (-1) * (J_to_eV) ** (-1) * eV ** (-1),
        hour: 60 * 60 * (hbar) ** (-1) * (J_to_eV) ** (-1) * eV ** (-1),
        day: 24 * 60 * 60 * (hbar) ** (-1) * (J_to_eV) ** (-1) * eV ** (-1),
        week: 7 * 24 * 60 * 60 * (hbar) ** (-1) * (J_to_eV) ** (-1) * eV ** (-1),
        year: 365 * 24 * 60 * 60 * (hbar) ** (-1) * (J_to_eV) ** (-1) * eV ** (-1),

        # mass
        kgrm: J_to_eV * c_light ** 2 * eV,
        grm: 10 ** (-3) * J_to_eV * c_light ** 2 * eV,

        msol: m_sol * J_to_eV * c_light ** 2 * eV
    }
}


def to_unit_sys(expr, unit_sys):
    """
    Convert expr to equivalent expression in unit_sys unit system.
    """
    return expr.subs(convert_dict[unit_sys])


def to_unit_sys_mag(expr, unit_sys):
    """
    Convert expr to equivalent expression in unit_sys unit system WITHOUT units.
    """

    new_expr = to_unit_sys(expr, unit_sys)

    # replace variables with 1
    sub_dict = {}
    for arg in new_expr.free_symbols:
        sub_dict[arg] = 1.0

    return float(new_expr.subs(sub_dict))


def to_unit_from_sys(expr, new_units, unit_sys):
    """
    Takes an expression in unit_sys unit system and tries to convert to new_units. Returns
    expr if the units are not equivalent.

    BUGS: unit_sys has to be defined by one unit (e.g. NateV)
    """

    new_units_sys = to_unit_sys(new_units, unit_sys)

    # check if conversion is possible
    if (1.0 * new_units_sys).args[1:] == (1.0 * expr).args[1:]:

        new_expr = expr / to_unit_sys(new_units, unit_sys)

        return new_expr * new_units

    else:

        print('Cannot convert units. Returning expr in natural (eV) units.')
        return expr


def to_unit(expr, new_units, base_unit_sys='NateV'):
    """
    Tries to convert expr to the equivalent expression with units new_units. Returns expr in
    natural units if the conversion is not possible.
    """
    return to_unit_from_sys(to_unit_sys(expr, base_unit_sys), new_units, base_unit_sys)


def to_unit_mag(expr, new_units, base_unit_sys='NateV'):
    new_expr = to_unit(expr, new_units, base_unit_sys)

    return float(new_expr.args[0])