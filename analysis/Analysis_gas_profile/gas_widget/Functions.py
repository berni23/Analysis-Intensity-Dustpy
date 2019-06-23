import matplotlib.pyplot as plt
import numpy as np
import glob
from matplotlib.widgets import Slider
import astropy.constants as u
from scipy.interpolate import interp2d


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

def sigma_unperturbed(r, Sigma_0_fact, rc, gamma_sigma):

    " Uperturbed gas profile"

    Sigma = Sigma_0_fact * (r / rc)**(-gamma_sigma) * np.exp(-(r / rc)**(2 - gamma_sigma))
    return Sigma


"reading in hydro data"
hydro_r = np.loadtxt('data_evolving/domain_y.dat')
files = sorted(glob.glob('data_evolving/*.txt'))
hydro_data = np.array([np.loadtxt(file) for file in files])
times = np.array([float(file[-11:-4]) for file in files])
times /= times[-1]

rc = 200 * c.au
hydro_T = 23 * (hydro_r / rc)**-0.3

f2d = interp2d(np.log10(hydro_r), times, np.log10(hydro_data))

def get_hydro(r, t):

    "Getting the hidrodynamical profile and interpolating for any radius and time"
    return 10**f2d(np.log10(r),t)

def kanagawa_time(r,r_p,M_p,t,l,sig_0,alpha,smooth,hp,Q=1):


    hp = np.interp(r_p, r, np.ones_like(r) * hp)


    def q(t,alpha):


        """Function for the evolution of the planet to star mass ratio over time. The parameters A,B have been obtained
        by using the depth to planet mass relation of the kanagawa profile  in one hand and the depth of the hydrodinamic profile over
        time on the other hand, assuming a final mass of Mp1,2,3 = 0.6,1,1.3 """

        C = 20*alpha*(hp/r_p)**5
        A,B=0,0

        if Q==1:

            "Q=1, parameters obtained for the most inner planet."

            A,B = 0.14473522, 0.3535564

        elif Q==2:

            "Q=2, parameters obtained for the middle planet."

            A,B =0.09207304, 0.43570505

        elif Q==3:

            "Q=3, parameters obtained for the most  outer planet."

            A,B = 0.02744491, 0.55246773

        elif Q==4: 

          "Q=4, parameters obtained  averaging for the three planets."

          A,B = 0.08808439, 0.44724306

        elif Q==5:

             "Q=5, Parameters fine tunned and using the viscous timescale from Kanagawa, allowing for planet mass variation."

             A,B =1.06527348, 0.44724306

             t = t/t_max(r_p, M_p,alpha)*1.4*1e3*c.yr

        return np.sqrt(C*(np.exp(A*(t*800)**B)-1))


    "Setting up the Kanagawa profile"
    r_H = r_p*(q(t,alpha)/3.)**(1./3.)

    k_prime =  (q(t,alpha))**2*(hp/r_p)**-3/alpha

    k1 = (q(t,alpha))**2*(hp/r_p)**-5/alpha

    sigma_min =  1/(1.+0.04*k1)

    delta_r1 = r_p*(sigma_min/4. + 0.08)*k_prime**0.25

    delta_r2 =  0.33*k_prime**0.25*r_p

    sigma_gap = 4*k_prime**-0.25*np.abs(r-r_p)/r_p - 0.32

    mask1 = np.abs(r - r_p) < delta_r1

    mask2 = (delta_r1 <= np.abs(r - r_p)) & (np.abs(r - r_p) <= delta_r2)

    sigma = np.ones_like(r)

    sigma[mask1] = sigma_min

    sigma[mask2] = sigma_gap[mask2]

    "Smoothing factor"

    # drsmooth = smooth * r_H
    # #
    # sigma = np.exp(np.log(sigma) * np.exp(-0.5 * (r - r_p)**4 / drsmooth**4))
    # print(np.sum(sigma))

    return sigma

def c_sound(r):


    return np.sqrt(((r / 200/c.AU)**(-0.3) * 23 * c.k) / c.mp / 2.3)

def omega(r):

    return np.sqrt(c.G * c.Mstar * r**(-3))

def H_r(r):

        return c_sound(r) / omega(r)/r


def t_max(r_p, M_p,alpha):

    "opening timescale , expression (4) Kanagawa et al 2017"

    return 1e6 * c.yr * 0.24 *(M_p /c.Mstar/ 1e-3) * (alpha/ 1e-3)**(-3 / 2) * (c.Mstar / c.Msun)**(-0.5) * (r_p / (10 / c.au))**(3 / 2) * (H_r(r_p) / 0.05)**(-7 / 2)


def get_fit(r,sig_0,t,masses,radii,S,hp,alpha,profile =1):

    "Function that fits the hydrodinamical profile with the help of Kanagawa time function."

    "profile =1, the parameters obtained for each planet for q(t) are used."
    "profile = 2, an average of A and B is used for the three planets."
    "profile = 3, a fine tunned verison of q(t) allowing for planet mass variation is used."

    l3 = 0.0
    l1 = 0.0
    sig_0 = sig_0.copy()

    if profile ==1:
      planet= [1,2,3]

    elif profile ==2:

      planet = [4,4,4]

    elif profile ==3:

      planet = [5,5,5]

    factor = np.ones_like(r)

    for mp, rp,qu in zip(masses, radii,planet):
            factor *=  kanagawa_time(r, rp * c.AU, mp * c.Mj, t,l1, sig_0,alpha,S,hp,Q=qu)

    return sig_0*factor
