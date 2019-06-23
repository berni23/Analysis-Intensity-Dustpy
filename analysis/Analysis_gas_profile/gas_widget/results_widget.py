import matplotlib.pyplot as plt
import numpy as np
import glob
from matplotlib.widgets import Slider
import astropy.constants as u
from scipy.interpolate import interp2d
from Functions import get_fit,c,get_hydro,hydro_r,hydro_data



"""widget for the comparison between different functions of  q(t) ( Mplanet/Mstar)"""



r = np.logspace(1, np.log10(300), 300) * c.au
sig_0 = get_hydro(r, 0)

# Make the figure
fig = plt.figure()
ax = fig.add_axes([0.1, 0.3, 0.8, 0.65])
l1, = ax.loglog(hydro_r/c.au , hydro_data[0], 'k-', label='hydro')
l2, = ax.loglog(r/c.au , sig_0, 'r', label='fit, q1')
l3, = ax.loglog(r/c.au , sig_0, 'g', label='fit, q2')
l4, = ax.loglog(r/c.au , sig_0, 'b', label='fit, q3')
ax.legend(fontsize='small')
ax.set_xlim(r[[0, -1]] / c.au)
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
axTime = plt.axes([x0 + 0.15 * width,
                   0.20,
                   0.75 * width,
                   0.05 * height], facecolor="lightgoldenrodyellow")

sliderTime = Slider(axTime, "Time", 0, 1, valinit=0, valfmt="%.02f")
title = axTime.set_title('time = ')
ax._widgets += [sliderTime]

# mass slider

axMass1 = plt.axes([x0 + 0.15 * width,
                    0.15,
                    0.25 * width,
                    0.05 * height], facecolor="lightgoldenrodyellow")

axMass2 = plt.axes([x0 + 0.15 * width,
                    0.10,
                    0.25 * width,
                    0.05 * height], facecolor="lightgoldenrodyellow")

axMass3 = plt.axes([x0 + 0.15 * width,
                    0.05,
                    0.25 * width,
                    0.05 * height], facecolor="lightgoldenrodyellow")


sliderMass1 = Slider(axMass1, "$M_1$", 0, 4, valinit=1.04, valfmt="%.2f")
sliderMass2 = Slider(axMass2, "$M_2$", 0, 4, valinit=1.72, valfmt="%.2f")
sliderMass3 = Slider(axMass3, "$M_3$", 0, 4, valinit=1.76, valfmt="%.2f")

ax._widgets += [sliderMass1]
ax._widgets += [sliderMass2]
ax._widgets += [sliderMass3]

axSmooth = plt.axes([x0 + 0.15*width,
                    0.01,
                    0.25,
                    0.05*height],facecolor = "lightgoldenrodyellow")

sliderSmooth = Slider(axSmooth, "$Smoothing$", 0, 10, valinit=3, valfmt="%.2f")

ax._widgets += [axSmooth]
# position slider

axRadius1 = plt.axes([x0 + 0.55 * width,
                      0.15,
                      0.25 * width,
                      0.05 * height], facecolor="lightgoldenrodyellow")

axRadius2 = plt.axes([x0 + 0.55 * width,
                      0.10,
                      0.25 * width,
                      0.05 * height], facecolor="lightgoldenrodyellow")

axRadius3 = plt.axes([x0 + 0.55 * width,
                      0.05,
                      0.25 * width,
                      0.05 * height], facecolor="lightgoldenrodyellow")

axAlpha = plt.axes([x0 + 0.55 * width,
                      0.01,
                      0.25 * width,
                      0.05 * height], facecolor="lightgoldenrodyellow")


sliderRadius1 = Slider(axRadius1, "$R_1$", 1, 300, valinit=65, valfmt="%.2f")
sliderRadius2 = Slider(axRadius2, "$R_2$", 1, 300, valinit=100, valfmt="%.2f")
sliderRadius3 = Slider(axRadius3, "$R_3$", 1, 300, valinit=163, valfmt="%.2f")

sliderAlpha = Slider(axAlpha, r'$\log\alpha$', -80, 20, valinit=-3.5, valfmt='%.2f')
ax._widgets += [sliderAlpha]
ax._widgets += [sliderRadius1]
ax._widgets += [sliderRadius2]
ax._widgets += [sliderRadius3]
# ax._widgets += [sliderBeta]

#
# axL= plt.axes([x0 + 0.85 * width,
#                       0.15,
#                       0.25 * width,
#                       0.05 * height], facecolor="lightgoldenrodyellow")


# sliderL = Slider(axL, "r$L$", 1, 10 ,valinit=0, valfmt="'%.2f")
#
# ax._widgets += [sliderL]

hydro_r = np.loadtxt('data_evolving/domain_y.dat')
files = sorted(glob.glob('data_evolving/*.txt'))
hydro_data = np.array([np.loadtxt(file) for file in files])
times = np.array([float(file[-11:-4]) for file in files])
times /= times[-1]

rc = 100*c.au
hydro_T = 24 * (hydro_r / rc)**-0.3
hydro_hp = np.sqrt(u.k_B.cgs.value * hydro_T * hydro_r**3 / (c.Mstar * u.m_p.cgs.value * u.G.cgs.value ))
r = np.logspace(1, np.log10(300), 300) * c.au

hp = 10.**np.interp(np.log10(r), np.log10(hydro_r), np.log10(hydro_hp))


def callback(event):

    global hp
    global r
    """
    The callback for updating the figure when the buttons are clicked
    """
    # get all values from the sliders

    time = sliderTime.val
    M1 = sliderMass1.val
    M2 = sliderMass2.val
    M3 = sliderMass3.val
    R1 = sliderRadius1.val
    R2 = sliderRadius2.val
    R3 = sliderRadius3.val
    S =  sliderSmooth.val
    Alpha = 10**sliderAlpha.val

    # print('alpha slider = '+ str(Alpha))


    # beta = sliderBeta.val
    # l = sliderL.val
    # update log sliders with a meaningful value

    title.set_text(f'time = {time:.2g}')

    # get the new profiles

    sig_hydro = get_hydro(hydro_r, time)
    #
    sig_fit = get_fit(r, sig_0, time,[M1, M2, M3],[ R1, R2, R3],S,hp,Alpha,profile =1)
    sig_fit2 = get_fit(r, sig_0, time,[M1, M2, M3],[ R1, R2, R3],S,hp,Alpha,profile =2)
    sig_fit3 = get_fit(r, sig_0, time,[M1, M2, M3],[ R1, R2, R3],S,hp,Alpha,profile =3)



    # update the lines with the new profiles

    l1.set_ydata(sig_hydro)
    l2.set_ydata(sig_fit)
    l3.set_ydata(sig_fit2)
    l4.set_ydata(sig_fit3)

    plt.draw()

# connect all sliders to the callback function

sliderTime.on_changed(callback)
sliderMass1.on_changed(callback)
sliderMass2.on_changed(callback)
sliderMass3.on_changed(callback)
sliderRadius1.on_changed(callback)
sliderRadius2.on_changed(callback)
sliderRadius3.on_changed(callback)
sliderSmooth.on_changed(callback)
sliderAlpha.on_changed(callback)
# sliderL.on_changed(callback)

# call the callback function once to make the plot agree with state of the buttons
callback(None)

plt.show()
