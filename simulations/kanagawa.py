import astropy.units as u
import astropy.constants as cc
from scipy import interpolate
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pickle

class c:

    rj = 71492e+5  # jupyter radius in cm
    au = 1.496e+13
    AU = au
    Msun = 1.98855e+33
    Mj = 1.898e+30
    Mstar = 2.3 * Msun
    G = 6.6725985e-8  # cm3 g-1 s-2
    alpha = 1e-3  # viscosity parameter
    k = 1.38064853e-16    # Boltzmann cnst (cgs)
    mp = 1.6726219e-24
    rc = 200.868 * au
    gamma = 0.8
    rin = 20 * au
    c_sound2 = (21 * k) * (2.3 * mp * rc**(-0.3))**(-1)
    yr = 3.154e+7
    pc = 3.086e+18

"""Functions needed for the implementation of the Kanagawa profile in dustpy """

def sigma_unperturbed(r, Sigma_0_fact, rc, gamma_sigma):

    """ Unperturbed gas surface density profile"""

    Sigma = Sigma_0_fact * (r / rc)**(-gamma_sigma) * np.exp(-(r / rc)**(2 - gamma_sigma))

    return Sigma

def c_sound(r,sim):

    """ Speed of sound"""

    return np.sqrt(((r / sim.ini.gas.TR0)**(-sim.ini.gas.TExp) * sim.ini.gas.T0 * c.k) / c.mp /2.3)

def omega(r):

    """Angular Speed"""

    return np.sqrt(c.G * c.Mstar * r**(-3))

def H_r(r, sim):

    """Scale height"""

    return (c_sound(r, sim)) / (r * omega(r))

def t_max(r_p, M_p, sim,alpha):

    """ Gap opening timescale , expression (4) Kanagawa et al 2017"""

    return 1e6 * c.yr * 0.24 *(M_p /c.Mstar/ 1e-3) * (alpha / 1e-3)**(-3 / 2) * (c.Mstar / c.Msun)**(-0.5) * (r_p / (10 * c.au))**(3 / 2) * (H_r(r_p, sim) / 0.05)**(-7 / 2)

def kanagawa_time(sim, r, r_p, M_p, t, l, Sigma_0_fact,rc,gamma,alpha,Q):

    """ Kanagawa profile, Kanagawa et al , 2016
    sim: simulation
    r : radial meshgrid
    r_p : planet Radius
    M_p: planet mass
    t : time
    Sigma_0_fact: Normalization factor for the unpertubed gas surface density Profile
    rc: cut off radius for the unperturbed gas surfacxe density profile
    gamma: exponential factor in the gas surface density Profile
    alpha: viscosity parameter

    Q : profile that will be used to parametrize the planet to star mass ratio,

        Q=0 : Profile of the form A*exp(tau-1)

        Q=1,2,3: profiles derived from the relation between the gap depths in the hydrodinamic simulation (Teague et al,2018)
        and the planet mass to depth  ratio ( kanagawa et al, 2016) 1 -Inner planet / 2 - Middle planet / 3 - Outer planet
    """

    s = 3.5

    def q(t,sim):

        t_opening = t_max(r_p, M_p, sim,alpha)

        if t <= 0:

            return 0

        if Q==0:

          if t / t_opening < 1:

            return (t /t_opening) / (t /t_opening + 1e-2) * np.exp(t /t_opening - 1) * M_p / (c.Mstar * (1 + l * t /t_opening))

          else:

            return M_p / c.Mstar

        else:
                    C = 20*alpha*(H_r(r_p,sim))**5

                    if Q==1:

                        A,B = 0.14473522, 0.3535564

                    elif Q==2:

                       A,B =0.09207304, 0.43570505

                    elif Q==3:

                       A,B = 0.02744491, 0.55246773

                    res = np.sqrt(C*(np.exp(A*(t*1.4*1e3*c.yr)**B)-1))

                    if res < M_p/c.Mstar:

                       return  res

                    else:
                       return M_p/c.Mstar

    r_H = r_p*(q(t,sim)/3.)**(1./3.)
    k_prime = lambda r: alpha**(-1)*q(t,sim)**2*(H_r(r_p,sim))**(-3)
    k1 = lambda r: alpha**(-1)*((q(t,sim))**2)*(H_r(r_p,sim))**(-5)
    sigma_min = lambda r: (1+0.04*k1(r))**(-1)
    sigma_0 = sigma_unperturbed(r,Sigma_0_fact,rc,gamma)
    delta_r1 = lambda r: r_p*(sigma_min(r)/4+ 0.08)*k_prime(r)**(0.25)
    delta_r2 = lambda r: r_p*0.33*k_prime(r)**(0.25)
    sigma_gap = 4.0*k_prime(r)**(-0.25)*np.abs(r-r_p)/r_p -0.32
    mask1 = np.array(np.abs(r - r_p) < delta_r1(r))
    mask2 = (delta_r1(r) <= np.abs(r - r_p)) & (np.abs(r - r_p) <= delta_r2(r))
    sigma = np.ones_like(r)
    sigma[mask1] = sigma_min(r[np.where(mask1 == True)])
    sigma[mask2] = sigma_gap[mask2]
    smooth = s
    drsmooth = smooth * r_H

    if t<=0:
        return np.ones_like(r)
    else:

         sigma = np.exp(np.log(sigma) * np.exp(-0.5 * (r - r_p)**4 / drsmooth**4))
         return sigma

def M_to_sigma(sim, Mgas, rc, gamma):

    "Transforming the Mgas input to the sigma factor for the unperturbed gas density profile"

    M_to_sigma = Mgas*c.Mstar/2/np.pi/np.trapz( sim.grid.rInt * sigma_unperturbed(sim.grid.rInt, 1, rc, gamma), x=sim.grid.rInt, dx=sim.grid.rInt[1] - sim.grid.rInt[0])

    return M_to_sigma

def kanagawa_time_3gap(sim, r,t, Mgas, tp2=0, tp3=0, Mp1=0.6, Mp2=1, Mp3=1.3, rp1=56, rp2=83, rp3=125, rc=100, gamma=0.2,alpha=1e-3,l3 = 0.2, profile='hydro'):

    "Function that uses the kanagawa profile defined above and multiplies it for the three planet cases. This function is called when performing the simulation in dustpy"

    l1 = 0
    M= M_to_sigma(sim,Mgas,rc*c.AU,gamma)

    if profile =='hydro':

        Q =[1,2,3]

    elif profile=='exp':

        Q =[0,0,0]

    return kanagawa_time(sim, r, rp1 * c.AU, Mp1 * c.Mj, t , l1, M, rc * c.AU, gamma,alpha,Q[0]) * kanagawa_time(sim, r, rp2 * c.AU, Mp2 * c.Mj, t-tp2*c.yr*1e6, l1, M, rc * c.AU, gamma,alpha,Q[1])\
          * kanagawa_time(sim, r, rp3 * c.AU, Mp3 * c.Mj, t - tp3 * c.yr*1e6, l3, M, rc * c.AU, gamma,alpha,Q[2]) * sigma_unperturbed(r, M, rc * c.AU, gamma)
