
import dustpy
from dustpy.plotting.plot import readFilesFromDir, readFieldFromFile
import h5py
from helper import planck_B_nu, convolve, get_dsharp_data
from scipy.interpolate import interp2d
import dsharp_opac as op
import dustpy
import astropy.constants as c
import astropy.units as u
au = c.au.cgs.value
import matplotlib.pyplot as plt
import numpy as np


def dens_to_int_v2(name,i_file,opacities='default_opacities_smooth.npz'):

    """Function that transfortms the dust surface density of a given simulation to the intensity one would spot
    if the source was at "dpc" parsecs away from the earth, for a given opacities and grain sizes

    The function returns:

    [0] Radial coordinate [arc-sec]
    [1] Azymutally averaged Intensity [Jy/Sr]
    [2] Convolved intensity with the beam used in isella et al,2016 [Jy/Sr]

    """

    dpc = 101.
    beam_as = (0.22, 0.15)
    rms_muJ = 8.9e-5

    beam_sas = np.pi * np.prod(beam_as) / (2*np.sqrt(2)*np.log(2.))
    beam_sr  = (beam_sas * u.arcsec**2).to('sr').value
    beam_cm  = np.sqrt(np.prod(beam_as)) * dpc * au
    I_rms    = (rms_muJ * u.Jy / beam_sr).cgs.value

    files = readFilesFromDir(name, 'data*.hdf5')

    sig_d = readFieldFromFile('dust/Sigma', files[i_file])[()]
    a = readFieldFromFile('dust/a', files[i_file])[0, :][()]
    r = readFieldFromFile('grid/r', files[i_file])[()]
    T = readFieldFromFile('gas/T', files[i_file])[()]

    r_as = np.arctan(r/dpc/c.pc.value/1e2)*(180/np.pi)*60**2

    with np.load(op.get_datafile(opacities)) as f:
        a_op  = f['a']
        lam   = f['lam']
        k_abs = f['k_abs']
        k_sca = f['k_sca']
        g     = f['g']
        rho_s = f['rho_s']


    "pick 2 wavelength and interpolate the opacities on those wavelengths"

    lam_mm = np.array([0.1, 0.13])
    nu = c.c.cgs.value / lam_mm

    f_int = interp2d(np.log10(a_op), np.log10(lam), np.log10(k_abs.T))
    k_abs = 10.**f_int(a, lam_mm)

    tau = (sig_d[:, None, :] * k_abs[None, ...]).sum(-1)
    I_nu = planck_B_nu(nu, T) * (1 - np.exp(-tau))

    I_conv = convolve(r, I_nu[:,1], beam_cm)
    return  r_as,I_nu[:,1],I_conv
