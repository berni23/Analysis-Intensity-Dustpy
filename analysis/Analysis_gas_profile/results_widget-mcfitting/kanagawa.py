import numpy as np
class c:

    rj = 71492e+5  # jupyter radius in cm
    au = 1.496e+13
    AU = au
    Msun = 1.98855e+33
    #Mgas = .09 * Msun
    Mj = 1.898e+30
    Mstar = 2.3 * Msun
    G = 6.6725985e-8  # cm3 g-1 s-2
    alpha = 1e-3  # viscosity parameter
    k = 1.38064853e-16    # Boltzmann cnst (cgs)
    mp = 1.6726219e-24
    rc = 200.868 * au
    gamma = 0.8
    rin = 20 * au
    #Sigma_0_fact = 0.440964656096
    c_sound2 = (21 * k) * (2.3 * mp * rc**(-0.3))**(-1)
    yr = 3.154e+7
    pc = 3.086e+18


def get_fit(r, sig_ini, t, masses, radii, alpha_out,alpha_in,hp, M_star):

    factor = np.ones_like(r)


    for mp, rp in zip(masses, radii):
        factor *= get_kanagawa_factor(r, hp, mp, rp, M_star, alpha_out, alpha_in ,smooth=None)

    return sig_ini * (factor * t**.2 + (1 - t**.2))


def alpha(r,alpha_out,alpha_in):


        R = 199 * c.AU
        R0 = 58.6 * c.AU

        return alpha_in * (1. - (1. - alpha_out / alpha_in) / 2.0 * (1.0 - np.tanh((r - R) / R0)))


def get_kanagawa_factor(r, hp, m_planet, a_planet, mstar, a_out,a_in, smooth=None):


    """Short summary.

    Parameters
    ----------
    r : array
        radial grid

    hp : float | array
        pressure scale height (float or array)

    m_planet : float
        planet mass

    a_planet : float
        planet semimajor axis

    mstar : float
        stellar mass

    alpha : float | array
        turbulence parameter

    Keywords
    --------

    smooth : None | float
        if float, smooth the profile over that many hill radii

    Returns
    -------
    array
        surface density reduction factor

    """
    hp = np.interp(a_planet, r, np.ones_like(r) * hp)


    # alpha = np.interp(a_planet, r, alpha * np.ones_like(r))

    K = (m_planet / mstar)**2 * (hp / a_planet)**-5 /1e-3/alpha(a_planet,a_out,a_in)
    Kp = (m_planet / mstar)**2 * (hp / a_planet)**-3 /alpha(a_planet,a_out,a_in)

    factor_min = 1. / (1. + 0.04 * K)  # Eq. 11

    delta_R_1 = (factor_min / 4. + 0.08) * Kp**0.25 * a_planet  # Eq. 8
    delta_R_2 = 0.33 * Kp**0.25 * a_planet  # Eq. 9

    factor_gap = 4 * Kp**-0.25 * np.abs(r - a_planet) / a_planet - 0.32  # Eq. 7

    # Eqn. 6

    factor = np.ones_like(r)

    mask1 = np.abs(r - a_planet) < delta_R_1
    mask2 = (delta_R_1 <= np.abs(r - a_planet)) & (np.abs(r - a_planet) <= delta_R_2)

    factor[mask1] = factor_min
    factor[mask2] = factor_gap[mask2]

    if smooth is not None:
        r_H = a_planet * (m_planet / (mstar * 3.))**(1. / 3.)
        drsmooth = smooth * r_H
        factor = np.exp(np.log(factor) * np.exp(-0.5 * (r - a_planet)**4 / drsmooth**4))

    return factor
