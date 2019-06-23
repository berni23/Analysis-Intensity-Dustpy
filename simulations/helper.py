import numpy as np
from astropy import units as u
from astropy import constants as c
import matplotlib.pyplot as plt


def planck_B_nu(freq, T):
    """
    Calculates the value of the Planck-Spectrum
    B(nu,T) of a given frequency nu and temperature T

    Arguments
    ---------
    nu : float or array
        frequency in 1/s or with astropy.units

    T: float
        temperature in K or in astropy.units

    Returns:
    --------
    B : float
        value of the Planck-Spectrum at frequency nu and temperature T
        units are using astropy.units if the input values use those, otherwise
        cgs units: erg/(s*sr*cm**2*Hz)

    """

    if isinstance(T, u.quantity.Quantity):
        use_units = True
    else:
        T = T * u.K
        use_units = False

    if not isinstance(freq, u.quantity.Quantity):
        freq *= u.Hz

    T = np.array(T.value, ndmin=1) * T.unit
    freq = np.array(freq.value, ndmin=1) * freq.unit

    f_ov_T = freq[np.newaxis, :] / T[:, np.newaxis]
    mx = np.floor(np.log(np.finfo(f_ov_T.ravel()[0].value).max))
    exp = np.minimum(f_ov_T * c.h / c.k_B, mx)
    exp = np.maximum(exp, -mx)

    output = 2 * c.h * freq**3 / c.c**2 / (np.exp(exp) - 1.0) / u.sr

    cgsunit = 'erg/(s*sr*cm**2*Hz)'
    if use_units:
        return output.to(cgsunit).squeeze()
    else:
        return output.to(cgsunit).value.squeeze()


def convolve(r, I_nu, beam_au, n_sigma=5, n_beam=40):
    beam_sigma = beam_au / (2 * np.sqrt(2 * np.log(2)))
    dx = n_sigma * beam_sigma / n_beam

    n = (r[-1] - r[0]) / dx
    r_lin = r[0] + np.arange(n) * dx

    # grid used to calculate beam

    r_beam = np.arange(-n_beam, n_beam + 1) * dx

    # make beam

    beam = np.exp(-r_beam**2 / (2 * beam_sigma**2))
    beam = beam / beam.sum()

    # interpolate

    I_lin = 10**np.interp(np.log10(r_lin), np.log10(r), np.log10(I_nu))

    # convolve

    I_conv = np.convolve(I_lin, beam, mode='same')

    # interpolate back

    return 10**np.interp(np.log10(r), np.log10(r_lin), np.log10(I_conv))


def get_dsharp_data(fname='HD163296.profile.txt', T_rms=0.27, lam_obs=0.125):
    """
    Keywords:
    ---------

    fname : str
        filename to read data from

    T_rms : float
        rms noise in brightness temperature from DSHARP I

    lam_obs : float
        observed wavelength in cm
    """
    data = np.loadtxt(fname)

    def I_nu_from_T_b(T_b):
        c_light = c.c.cgs.value

        nu_obs  = c_light / lam_obs
        return 2 * nu_obs**2 * c.k_B.cgs.value * T_b / c_light**2

    # radius in arcseconds

    r_as = data[:, 1]

    # intensity in brightness temperature

    T_b   = data[:, 4]
    dT_b  = data[:, 5]  # uncertainty on T_b

    # convert to intensity in CGS
    I_rms   = I_nu_from_T_b(T_rms)
    I_nu    = I_nu_from_T_b(T_b)
    dI_nu   = I_nu_from_T_b(dT_b)

    f, ax = plt.subplots()
    ax.semilogy(r_as, I_nu)
    ax.fill_between(r_as, I_nu - dI_nu, I_nu + dI_nu, color='r', alpha=0.5)
    ax.axhline(I_rms, c='k', ls='--')
    ax.set_xlim(0, 2)
    ax.set_ylim(0.5 * I_rms, 1e-12)
    ax.set_xlabel(r'$radius$ [arcsec]')
    ax.set_ylabel(r'$I_\nu$ [erg / (s cm$^2$ Hz sr)]')

    return r_as, I_nu, dI_nu
