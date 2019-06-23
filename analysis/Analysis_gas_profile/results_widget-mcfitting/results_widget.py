import glob
import numpy as np
from scipy.interpolate import interp2d
from scipy.optimize import curve_fit
import astropy.constants as c
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

import emcee
import corner
from kanagawa import get_fit

" Montecarlo routine for the fitting of the hydrondinamical profile HD 163296"

au = c.au.cgs.value
M_Jup = c.M_jup.cgs.value

"read in hydro data"

hydro_r = np.loadtxt('data_evolving/domain_y.dat')
files = sorted(glob.glob('data_evolving/*.txt'))
hydro_data = np.array([np.loadtxt(file) for file in files])
times = np.array([float(file[-11:-4]) for file in files])
times /= times[-1]

"set other parameters"

alpha = 1e-3
rc = 200 * au
hydro_T = 23 * (hydro_r / rc)**-0.3
M_star = 2.3*c.M_sun.cgs.value
hydro_hp = np.sqrt(c.k_B.cgs.value * hydro_T * hydro_r**3 / (c.m_p.cgs.value * c.G.cgs.value * M_star))

"define hydro interpolation function"

f2d = interp2d(np.log10(hydro_r), times, np.log10(hydro_data))


def get_hydro(r, t):
    """
    Get hydro profile by interpolating the data.

    r : array
        radial array to interpolate onto

    t : float
        time at which the profile is interpolated

    Output
    ------

    array: gas surface density on hydro grid
    """
    return 10**f2d(np.log10(r), t)


"arbitrary initial condition"

r = np.logspace(1, np.log10(300), 300) * au
sig_0 = get_hydro(r, 0)

"interpolate scale height onto our grid"

hp = 10.**np.interp(np.log10(r), np.log10(hydro_r), np.log10(hydro_hp))

"try and fit the parameters"


def fit_wrapper(x, M1, M2, M3, R1, R2, R3, alpha_out,alpha_in):
    M1 *= M_Jup
    M2 *= M_Jup
    M3 *= M_Jup
    R1 *= au
    R2 *= au
    R3 *= au
    return get_fit(x, sig_0, 1.0, [M1, M2, M3], [R1, R2, R3], alpha_out,alpha_in, hp, M_star)

sig_goal = get_hydro(r, 1.0)

"try a local maximum approach"

# p0 = [0.80, 1.30, 1.45, 65, 102, 165, 1e-3]
p0 = [1.0, 1.0, 1.0, 65, 105, 161, 1e-3,1e-3]
bounds = (
    [0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1e-5,1e-5],
    [5, 5, 5, 300, 300, 300, 1e-1,1e-1])
params = curve_fit(fit_wrapper, r, sig_goal, sigma=sig_goal / 10.0, p0=p0, bounds=bounds)[0]

print(params)

"try a mc-approach"


def lnprob(params, r, sig_goal):
    """
    return a log-probability value for the given parameters and fit values

    Parameters
    ----------

    params : array
        7 element array with the 3 masses, 3 positions, and alpha

    r, sig_goal : arrays
        radial and surface density array

    Returns
    -------
    float
        log-probability value
    """
    params = np.array(params)
    params[-1] = 10.**params[-1]
    params[-2] = 10.**params[-2]

    r1, r2, r3 = params[3:6]
    if not r1 < r2 < r3:
        return -np.inf

    if params[-1] > 1.0:
        return -np.inf

    if np.any(params < bounds[0]) or np.any(params > bounds[1]):
        return -np.inf

    fit = fit_wrapper(r, *params)
    return -np.sum((fit - sig_goal)**2 / (0.1 * sig_goal)**2)


ndim = len(params)
nwalkers = 500
nburn = 5000
nsteps = 50000
ini_walkers = [[lower + (upper - lower) * np.random.rand() for upper, lower in np.array(bounds).T[:-1]] for i in range(nwalkers)]
ini_walkers = np.hstack((np.array(ini_walkers), (-5 + 4 * np.random.rand(nwalkers))[:, None]))
ini_walkers[:, 3:6] = np.sort(ini_walkers[:, 3:6])

# burn-in and run

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[r, sig_goal])
pos_b, prob_b, state_b = sampler.run_mcmc(ini_walkers, nburn)
sampler.reset()
pos, prob, state = sampler.run_mcmc(pos_b, nsteps)
ch = sampler.chain.reshape(nwalkers * nsteps, ndim)
f = corner.corner(ch)
f.savefig('corner.pdf')
params_mc = pos[prob.argmax(), :]
params_mc[-2] = 10**params_mc[-2]
params_mc[-1] = 10**params_mc[-1]


print(params_mc)
# plot the results

f, ax = plt.subplots()
ax.loglog(r / au, sig_goal, 'r', label='hydro')
ax.loglog(r / au, fit_wrapper(r, *p0), '0.5', label='guess')
ax.loglog(r / au, fit_wrapper(r, *params), 'k--', label='fit')
ax.loglog(r / au, fit_wrapper(r, *params_mc), 'b--', lw=3, label='mcmc fit')
ax.set_ylim(1e-2, 1e2)
ax.set_xlabel(r'$r [au]$')
ax.set_ylabel(r'$\Sigma_\mathrm{g}$')
ax.legend()

# Make the figure

fig = plt.figure(figsize=(6, 6))
ax = fig.add_axes([0.1, 0.4, 0.8, 0.55])
l1, = ax.loglog(hydro_r / au, hydro_data[0], 'k-', label='hydro')
l2, = ax.loglog(r / au, sig_0, 'r', label='fit')
ax.legend(fontsize='small')
ax.set_xlim(r[[0, -1]] / au)
ax.set_ylim(1e-3, 1e3)

# to avoid garbage collection

ax._widgets = []

x0 = ax.get_position().x0
x1 = ax.get_position().x1
y0 = ax.get_position().y0
y1 = ax.get_position().y1
width = x1 - x0
height = y1 - y0

# CREATE SLIDERS

# time slider
axTime = plt.axes([x0, 0.30, width, 0.05 * height], facecolor="lightgoldenrodyellow")
sliderTime = Slider(axTime, "time", 0, 1, valinit=1, valfmt="%.2f")
ax._widgets += [sliderTime]

# mass slider

axMass1 = plt.axes([x0, 0.25, 0.3 * width, 0.05 * height], facecolor="lightgoldenrodyellow")
axMass2 = plt.axes([x0, 0.20, 0.3 * width, 0.05 * height], facecolor="lightgoldenrodyellow")
axMass3 = plt.axes([x0, 0.15, 0.3 * width, 0.05 * height], facecolor="lightgoldenrodyellow")

sliderMass1 = Slider(axMass1, "$M_1$", 0.1, 5, valinit=0.80, valfmt="%.2f MJup")
sliderMass2 = Slider(axMass2, "$M_2$", 0.1, 5, valinit=1.30, valfmt="%.2f MJup")
sliderMass3 = Slider(axMass3, "$M_3$", 0.1, 5, valinit=1.45, valfmt="%.2f MJup")

ax._widgets += [sliderMass1]
ax._widgets += [sliderMass2]
ax._widgets += [sliderMass3]

# position slider

axRadius1 = plt.axes([x0 + 0.5 * width, 0.25, 0.3 * width, 0.05 * height], facecolor="lightgoldenrodyellow")
axRadius2 = plt.axes([x0 + 0.5 * width, 0.20, 0.3 * width, 0.05 * height], facecolor="lightgoldenrodyellow")
axRadius3 = plt.axes([x0 + 0.5 * width, 0.15, 0.3 * width, 0.05 * height], facecolor="lightgoldenrodyellow")

sliderRadius1 = Slider(axRadius1, "$R_1$", 1, 300, valinit=65, valfmt="%.2f au")
sliderRadius2 = Slider(axRadius2, "$R_2$", 1, 300, valinit=102, valfmt="%.2f au")
sliderRadius3 = Slider(axRadius3, "$R_3$", 1, 300, valinit=165, valfmt="%.2f au")

ax._widgets += [sliderRadius1]
ax._widgets += [sliderRadius2]
ax._widgets += [sliderRadius3]

# alpha slider

axAlpha = plt.axes([x0, 0.10, 0.3 * width, 0.05 * height], facecolor='lightgoldenrodyellow')
sliderAlpha = Slider(axAlpha, r'$\log\alpha_{in}$', -4, -1, valinit=-3, valfmt='%.2f')
ax._widgets += [sliderAlpha]

axAlpha_out = plt.axes([x0, 0.05, 0.3 * width, 0.05 * height], facecolor='lightgoldenrodyellow')
sliderAlpha_out = Slider(axAlpha_out, r'$\log\alpha{out}$', -4, -1, valinit=-3, valfmt='%.2f')
ax._widgets += [sliderAlpha_out]


def callback(event):
    """
    The callback for updating the figure when the buttons are clicked
    """

    # get all values from the sliders

    time = sliderTime.val
    M1 = sliderMass1.val * M_Jup
    M2 = sliderMass2.val * M_Jup
    M3 = sliderMass3.val * M_Jup
    R1 = sliderRadius1.val * au
    R2 = sliderRadius2.val * au
    R3 = sliderRadius3.val * au
    alpha_in = 10**sliderAlpha.val
    alpha_out = 10**sliderAlpha_out.val

    # get the new profiles

    sig_hydro = get_hydro(hydro_r, time)
    sig_fit = get_fit(r, sig_0, time, [M1, M2, M3], [R1, R2, R3], alpha_in,alpha_out, hp, M_star)

    # update the lines with the new profiles

    l1.set_ydata(sig_hydro)
    l2.set_ydata(sig_fit)

    plt.draw()


# connect all sliders to the callback function

sliderTime.on_changed(callback)
sliderMass1.on_changed(callback)
sliderMass2.on_changed(callback)
sliderMass3.on_changed(callback)
sliderRadius1.on_changed(callback)
sliderRadius2.on_changed(callback)
sliderRadius3.on_changed(callback)
sliderAlpha.on_changed(callback)
sliderAlpha_out.on_changed(callback)

# call the callback function once to make the plot agree with state of the buttons
callback(None)

plt.show()
