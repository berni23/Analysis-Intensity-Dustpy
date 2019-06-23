
import matplotlib.pyplot as plt
import matplotlib.widgets as w
import numpy as np
import decimal
import pickle
import h5py
import glob
import os
import seaborn as sns
import astropy
import sys

from astropy import constants as u
from matplotlib.pyplot import rcParams
from matplotlib import colors as mcolors
from itertools import compress


# Path to the simulations
Path = '../simulations'
sys.path.insert(0,Path)
from dens_v2 import dens_to_int_v2
from sim_functions import c,Video


"""

Interface for the visualization of the  convolved instensity for a given frequency and opacities, by Bernat Ferrer and Til Birnstiel.
To include an extra variable, look for all the places were "example" is written, and substitute it for the desired variable.
Compilation : "python -B interface.py"

"""

rcParams['font.family'] = 'italic'

plt.style.use(['seaborn', {'figure.dpi': 100}])

my_colors = sns.color_palette(n_colors = 1000)

data = []

names = np.asarray([str(f) for f in glob.glob('../simulations/post*')])

# List of folders not to be considered as a simulations, now just folders starting with "post" are being considered
# No_sim = [str(Path)+'__pycache__',str(Path)+'data_evolving',str(Path)+'opacities']

"""Loop for downloading the data. For that , the simulations should include a dile I_R_v2.txt with the
radial coordinate and the corresponding intensity, to get it, run "set_sim_files" in the folder were the simulations are contained"""

for i in names:

    if os.path.isdir(str(i)) == True:

     try:
        with h5py.File(str(i) + '/data0050.hdf5') as f:

            r_AU = np.linspace(2,500,301)*c.au

            #r_AU = f['grid/r'][()]
            Texp = f['ini/gas/TExp'][()]
            TR0 = f['ini/gas/TR0'][()]
            cs = f['gas/cs'][()][()]
            vf = f['dust/vFrag'][()][0]
            t = float(f['t'][()]) / 1e6 / c.yr
            """"example = f['path/to/variable'][()]"""

        """Some variables are stored by default in hdf5 files, but others like planet mass are not,
        those are stored in Extra_vars. To add more variables , modify the class in sim_functions"""

        with open(str(i) + '/Extra_vars.pickle', 'rb') as dD:
            P = pickle.load(dD)

        r_Angle, Inu = np.loadtxt(str(i) + '/I_R_v2', comments='#', delimiter=',') # Dsharp
        r_Angle, Inu_op1 = np.loadtxt(str(i) + '/I_R_v1', comments='#', delimiter=',') # Ricci


        data.append([
            i,
            np.around(P.Mp1,decimals=0),
            np.around(P.Mp2,decimals=0),
            np.around(P.Mp3,decimals=0),
            P.rp1,
            P.rp2,
            P.rp3,
            np.around(P.alpha,decimals=5),
            P.gamma,
            np.around(P.Mgas, decimals=3),
            np.around(P.tp2 / 1e6, decimals=5),
            np.around(P.tp3 / 1e6, decimals=5),
            Texp,
            np.around(TR0 / c.au, decimals=2),
            np.around(t, decimals=2),
            np.around(vf, decimals=1),
            #example,
            r_Angle,
            Inu_op1,
            Inu
        ])

     except:

        print(str(i) +' is probably not a simulation, or it did not finish.')

"""Set the main screen"""

fig = plt.figure()
ax = fig.add_axes([0.2, 0.3, 0.6, 0.6])
plt.subplots_adjust(left=0.25)

ax.set_ylabel(r'$I_{\nu}$ [Jy/Sr] ', axes=ax, fontsize=8)
ax.set_xlabel(r'$\theta$ [arcsec]', axes=ax, fontsize=8)
ax.set_ylim(1e-20,1e-10)
ax.set_xlim(0,3)

"""Profile dsharp"""
HD = np.genfromtxt('HD163296.profile.txt')
lam_obs= 0.13

r_HD_au =  HD[:, 0]
I_HD = HD[:, 4] * 2 * lam_obs**-2 * u.k_B.cgs.value # this should be in cgs
# sigma_HD = HD[:,5]*2 * lam_obs**-2 * cc.k_B.cgs.value
theta_HD = np.arctan(HD[:, 0]*c.au/101.5/c.pc) * 60**2 * (180 / np.pi)

"""isella profile"""

R,I = np.loadtxt('isella.txt')

ax.plot(R,I,label ='Isella profile')
ax.plot(theta_HD,I_HD,label ='Andrews profile')

ax.set_xscale('linear')
ax.set_yscale('log')
ax.legend()

"""Sor all the values the variables take from smaller to bigger"""

name_values = sorted(list(set([d[0] for d in data])))
Mp1_values = sorted(list(set([d[1] for d in data])))
Mp2_values = sorted(list(set([d[2] for d in data])))
Mp3_values = sorted(list(set([d[3] for d in data])))
rp1_values = sorted(list(set([d[4] for d in data])))
rp2_values = sorted(list(set([d[5] for d in data])))
rp3_values = sorted(list(set([d[6] for d in data])))
alpha_values = sorted(list(set([d[7] for d in data])))
gamma_values = sorted(list(set([d[8] for d in data])))
Mgas_values = sorted(list(set([d[9] for d in data])))
tp2_values = sorted(list(set([d[10] for d in data])))
tp3_values = sorted(list(set([d[11] for d in data])))
Texp_values = sorted(list(set([d[12] for d in data])))
TR0_values = sorted(list(set([d[13] for d in data])))
t_values = sorted(list(set([d[14] for d in data])))
vf_values = sorted(list(set([d[15] for d in data])))

"""example_values = sorted(list(set([d[position] for d in data])))"""

""" Number of permutaitons available"""

Num_permutations = len(Mp1_values)*len(Mp2_values)*len(Mp3_values)*len(rp1_values)*len(rp2_values)*len(rp3_values)*len(alpha_values)*len(gamma_values)*len(Mgas_values)*len(tp2_values)*len(tp3_values)*len(Texp_values)*len(TR0_values)*len(t_values)*len(vf_values)

""" For each variable, we add an option All to select all values at on_clicked """

name_values.append(str('All'))
Mp1_values.append(str('All'))
Mp2_values.append(str('All'))
Mp3_values.append(str('All'))
rp1_values.append(str('All'))
rp2_values.append(str('All'))
rp3_values.append(str('All'))
alpha_values.append(str('All'))
gamma_values.append(str('All'))
Mgas_values.append(str('All'))
tp2_values.append(str('All'))
tp3_values.append(str('All'))
Texp_values.append(str('All'))
TR0_values.append(str('All'))
t_values.append(str('All'))
vf_values.append(str('All'))

"""example_values.append(str('All'))"""

"""varible color for the BUTTONS"""

color ='white'

ax._widgets = []

"""CREATE BUTTONS"""

button_width = 0.07
button_lheigh = 0.06
button_x0 = 0.02
button_y0 = 0.5

"""Panel Control"""

figB = plt.figure(figsize=(15, 6))

n_buttons_1 = len(Mp1_values)

ax_Mp1 = figB.add_axes([button_x0 + 0 * button_width,  button_y0,
                         button_width, n_buttons_1 * button_lheigh], facecolor=color)

Mp1_buttons = w.CheckButtons(ax_Mp1, [s for s in Mp1_values],  [
                             False] + (n_buttons_1 - 1) * [False])
ax_Mp1.set_title(r'$M_\mathrm{p1} (M_j)$')
ax._widgets += [Mp1_buttons]
#
n_buttons = len(Mp2_values)
ax_Mp2 = figB.add_axes([button_x0 + 1* button_width, (n_buttons_1 - n_buttons) *
                        button_lheigh + button_y0,  button_width, n_buttons * button_lheigh], facecolor=color)
Mp2_buttons = w.CheckButtons(ax_Mp2, [s for s in Mp2_values],  [
                             False] + (n_buttons - 1) * [False])
ax_Mp2.set_title(r'$M_\mathrm{p2} (M_j)$')
ax._widgets += [Mp2_buttons]

n_buttons = len(Mp3_values)
ax_Mp3 = figB.add_axes([button_x0 + 2* button_width, (n_buttons_1 - n_buttons)
                        * button_lheigh + button_y0, button_width, n_buttons * button_lheigh], facecolor=color)
Mp3_buttons = w.CheckButtons(ax_Mp3, [s for s in Mp3_values],  [
                             False] + (n_buttons - 1) * [False])
ax_Mp3.set_title(r'$M_\mathrm{p3} (M_j)$')
ax._widgets += [Mp3_buttons]


"""Buttons corresponding to variables not being used"""

####################################

# n_buttons = len(snaphot)
# snapshot_buttons = w.CheckButtons(ax_)

# n_buttons = len(rp1_values)
# ax_rp1 = figB.add_axes([button_x0 +3* button_width,  (n_buttons_1 - n_buttons)
#                         * button_lheigh + button_y0,  button_width, n_buttons * button_lheigh], facecolor=color)
# rp1_buttons = w.CheckButtons(ax_rp1, [s for s in rp1_values],  [
#                              False] + (n_buttons - 1) * [False])
# ax_rp1.set_title(r'$r_\mathrm{p1} (AU)$')
# ax._widgets += [rp1_buttons]
#
# n_buttons = len(rp2_values)
# ax_rp2 = figB.add_axes([button_x0 + 4 * button_width,  (n_buttons_1 - n_buttons)
# * button_lheigh + button_y0, button_width, n_buttons * button_lheigh], facecolor=color)
# rp2_buttons = w.CheckButtons(ax_rp2, [s for s in rp2_values],  [
#                              False] + (n_buttons - 1) * [False])
# ax_rp2.set_title(r'$r_\mathrm{p2} (AU)$')
# ax._widgets += [rp2_buttons]
#
# n_buttons = len(rp3_values)
# ax_rp3 = figB.add_axes([button_x0 + 5 * button_width, (n_buttons_1 - n_buttons) *
#                         button_lheigh + button_y0, button_width, n_buttons * button_lheigh], facecolor=color)
# rp3_buttons = w.CheckButtons(ax_rp3, [s for s in rp3_values],  [
#                              False] + (n_buttons - 1) * [False])
# ax_rp3.set_title(r'$r_\mathrm{p3} (AU)$')
#ax._widgets += [rp3_buttons]

#######################

#
n_buttons = len(alpha_values)
ax_alpha = figB.add_axes([button_x0 + 3 * button_width, (n_buttons_1 - n_buttons) *
                          button_lheigh + button_y0, button_width, n_buttons * button_lheigh], facecolor=color)
alpha_buttons = w.CheckButtons(
    ax_alpha, [str(s) for s in alpha_values], (n_buttons) * [False])
ax_alpha.set_title(r'$\alpha$ ')
ax._widgets += [alpha_buttons]

n_buttons = len(gamma_values)
ax_gamma = figB.add_axes([button_x0 + 4 * button_width, (n_buttons_1 - n_buttons)
                          * button_lheigh + button_y0, button_width, n_buttons * button_lheigh], facecolor=color)
gamma_buttons = w.CheckButtons(ax_gamma, [str(s) for s in gamma_values],  [
                               False] + (n_buttons - 1) * [False])
ax_gamma.set_title(r'$\gamma$')
ax._widgets += [gamma_buttons]

n_buttons = len(Mgas_values)
ax_Mgas = figB.add_axes([button_x0 + 5 * button_width, (n_buttons_1 - n_buttons)
                         * button_lheigh + button_y0, button_width, n_buttons * button_lheigh], facecolor=color)
Mgas_buttons = w.CheckButtons(ax_Mgas, [str(s) for s in Mgas_values],  [
                              False] + (n_buttons - 1) * [False])
ax_Mgas.set_title(r'$M_\mathrm{gas} (M_{star})$')
ax._widgets += [Mgas_buttons]

n_buttons = len(Texp_values)
ax_Texp = figB.add_axes([button_x0 + 6* button_width, (n_buttons_1 - n_buttons) *
                         button_lheigh + button_y0, button_width, n_buttons * button_lheigh], facecolor=color)
Texp_buttons = w.CheckButtons(ax_Texp, [str(s) for s in Texp_values],  [
                              False] + (n_buttons - 1) * [False])
ax_Texp.set_title(r'$T_\mathrm{exp}$')
ax._widgets += [Texp_buttons]

n_buttons = len(vf_values)

ax_vf = figB.add_axes([button_x0 + 7* button_width, (n_buttons_1 - n_buttons) *
                         button_lheigh + button_y0, button_width, n_buttons * button_lheigh], facecolor=color)
vf_buttons = w.CheckButtons(ax_vf, [str(s) for s in vf_values],  [
                              False] + (n_buttons - 1) * [False])

ax_vf.set_title(r'$v_\mathrm{frag}$')

ax._widgets += [vf_buttons]

n_buttons = len(tp2_values)
ax_tp2 = figB.add_axes([button_x0 + 8* button_width,  (n_buttons_1 - n_buttons)
                        * button_lheigh + button_y0, button_width, n_buttons * button_lheigh], facecolor=color)
tp2_buttons = w.CheckButtons(ax_tp2, [s for s in tp2_values],  [
                             False] + (n_buttons - 1) * [False])
ax_tp2.set_title(r'$t_\mathrm{p2} (Myr)$')
ax._widgets += [tp2_buttons]

n_buttons = len(tp3_values)
ax_tp3 = figB.add_axes([button_x0 + 9 * button_width,  (n_buttons_1 - n_buttons)
                        * button_lheigh + button_y0, button_width, n_buttons * button_lheigh], facecolor=color)
tp3_buttons = w.CheckButtons(ax_tp3, [s for s in tp3_values],  [
                             False] + (n_buttons - 1) * [False])
ax_tp3.set_title(r'$t_\mathrm{p3 } (Myr) $')
ax._widgets += [tp3_buttons]

##############3

# n_buttons = len(TR0_values)
# ax_TR0 = figB.add_axes([button_x0 + 12* button_width, (n_buttons_1 - n_buttons) *
#                         button_lheigh + button_y0, button_width, n_buttons * button_lheigh], facecolor=color)
# TR0_buttons = w.CheckButtons(ax_TR0, [str(s) for s in TR0_values],  [
#                              False] + (n_buttons - 1) * [False])
# ax_TR0.set_title(r'$T_{R0} (AU)$')
# ax._widgets += [TR0_buttons]

#####################

n_buttons = len(t_values)
ax_t = figB.add_axes([button_x0 +10*button_width ,  (n_buttons_1 - n_buttons) *
                   button_lheigh + button_y0, button_width, n_buttons * button_lheigh], facecolor=color)
t_buttons = w.CheckButtons(ax_t, [s for s in t_values], [
                           False] + (n_buttons - 1) * [False])

ax_t.set_title(r'$time$ (Myr)')
ax._widgets += [t_buttons]


"""

Create your own button from the variable "example"

n_buttons =len(example_values)

ax_example = figB.add_axes([button_x0 +11*button_width ,  (n_buttons_1 - n_buttons) *
                   button_lheigh + button_y0, button_width, n_buttons * button_lheigh], facecolor=color)
example_buttons = w.CheckButtons(ax_example, [s for s in example_values], [
                           False] + (n_buttons - 1) * [False])

ax_example.set_title(r'$example$ (units)')
ax._widgets += [example_buttons]

"""

""" Buttons main screen"""

ax_Names = fig.add_axes([0.82,0.43,0.1,0.12],facecolor = color)
Names_buttons = w.CheckButtons(ax_Names,[str('Print names sim ')],[False])

ax_allSim = fig.add_axes([0.82, 0.83, 0.1, 0.12], facecolor=color)
allSim_buttons = w.CheckButtons(ax_allSim, [str('Show All')], [False])

ax._widgets += [allSim_buttons]

ax_refresh = fig.add_axes([0.82, 0.73, 0.1, 0.12], facecolor=color)
refresh_buttons = w.CheckButtons(ax_refresh, [str('Refresh')], [False])

ax._widgets += [refresh_buttons]

ax_legend = fig.add_axes([0.82, 0.63, 0.1, 0.12],facecolor = color)
legend_buttons = w.CheckButtons(ax_legend,[str('Legend')],[False])

ax._widgets += [legend_buttons]

ax_videos = fig.add_axes([0.82,0.53,0.1,0.12],facecolor = color)
video_buttons = w.CheckButtons(ax_videos,[str('Show video')],[False])

ax._widgets += [video_buttons]

ax_op = fig.add_axes([0.1, 0.1, 0.1, 0.1], facecolor='lightblue')
op_buttons = w.CheckButtons(
    ax_op, [str('Ricci')], [False])
ax_op.set_title('opacities')

op_values = ['Ricci']

ax._widgets += [op_buttons]

ax_docu = fig.add_axes([0.82,0.33,0.1,0.12],facecolor = color)
docu_buttons = w.CheckButtons(ax_docu,[str('Documentation')],[False])

ax._widgets += [docu_buttons]

fig2 = plt.figure()

def Plot(d,icol,k):

    """ Plot function"""

    ax.plot(d[-3], d[k], '0.5', picker =d[0], label='sim_name=' +str(d[0][15:-1]) +str(d[0][-1])+'\n $M_{p1}$  ='+str(d[1]) +'\n $M_{p2}$ = '+ str(d[2]) +'\n $M_{p3}$= '+ str(d[3])+\
    '\n $r_{p1}$= '+ str(d[4])+'\n $r_{p2}$ ='+ str(d[5]) +'\n $r_{p3}$ =' + str(d[6])+'\n' + r'$\alpha$ = '+str(d[7])+'\n'  + r'$\gamma$  = '+ str(d[8]) +'\n $M_{g}$ =' + str(d[9])+\
    '\n $t_{p2}$ =' + str(d[10]) + '\n $t_{p3}$ =' + str(d[11]) +'\n $Texp$ =' + str(d[12]) + '\n $TR_{0}$ = ' + str(d[13]) + \
    '\n $t$ ='+str(d[14]), c=my_colors[icol])

    plt.draw()

def conditions(event):

    """Function that links the buttons clicked by the user with the simulations that should be appearing on the screen"""

    global NameVideo

    k = -1
    Nsim = 0
    arr =[]

    for _line in ax.get_lines()[2:]:

        _line.remove()

    ax.set_prop_cycle(None)

    for icol, d in enumerate(data):

        if str('Ricci') in  op_selected:
            k = -2

        if allSim_states[0] == True:

            """Plot all the simulations if the button "allSim" is pressed"""

            Plot(d,icol,k)
            Nsim = len(data)
            arr.append(d[0])

        elif d[1] in Mp1_selected or str('All') in Mp1_selected:
            if d[2] in Mp2_selected or str('All') in Mp2_selected:
                if d[3] in Mp3_selected or str('All') in Mp3_selected:
                    # if d[4] in rp1_selected or str('All') in rp1_selected:
                    #     if d[5] in rp2_selected or str('All') in rp2_selected:
                    #         if d[6] in rp3_selected or str('All') in rp3_selected:
                                if d[7] in alpha_selected or str('All') in alpha_selected:
                                    if d[8] in gamma_selected or str('All') in gamma_selected:
                                         if d[9] in Mgas_selected or str('All') in Mgas_selected:
                                               if d[10] in tp2_selected or str('All') in tp2_selected:
                                                      if d[11] in tp3_selected or str('All') in tp3_selected:
                                                            if d[12] in Texp_selected or str('All') in Texp_selected:
                                                                #if d[13] in TR0_selected or str('All') in TR0_selected:
                                                                    if d[14] in t_selected or str('All') in t_selected:

                                                                        if d[15] in vf_selected or str('All') in vf_selected:

                                                                            """ if d[position] in example_selected or str('All') in example_selected: """

                                                                            Plot(d,icol,k)
                                                                            Nsim =Nsim +1
                                                                            arr.append(d[0])


        if legend_states[0] == True:

           """Show the legend of the simulations if the legend button on the main screen is pressed"""
           ax.legend()
        fig.canvas.mpl_connect('pick_event', line_picker)
        plt.draw()

    if docu_states[0] == True:

         Docu(event)

    if Names_states[0] == True:

            """ Show the number of simulations appearing on the screen, giving the user an idea of how much
            populated is the parameter space selected"""

            print('Number of simulations showing:'+str(Nsim))
            print('Names of Current Simulations : \n ')
            for j in range(len(arr)):
              print('\n'+str(arr[j]))

            print(str(Num_permutations)+' possible permutations with the data present,' +str(Num_permutations-len(name_values))+' of those still left to be done.')


    if video_states[0] ==True:

            try :

                film = str(Path)+ '/video/'+str(NameVideo[15:-1])+str(NameVideo[-1])+'.mp4'

                Video(film)

            except:

                print('No video for sim' +str(NameVideo))

def line_picker(event):

    """function activated when picking a concrete simulation. It plots a contour of the radius vs particle mass"""

    global fig2
    global NameVideo

    NameVideo = event.artist.get_picker()
    print(NameVideo)

    plt.close(fig2)
    fig2 = plt.figure()
    ax2 = fig2.add_axes([0.1, 0.1, 0.5, 0.5])

    fig2.text(0.7, 0.3, str(event.artist.get_label()), bbox=dict(
        facecolor='gold', edgecolor='black', boxstyle='round,pad=1'))

    """Contour plot with the variables of the simulations written down"""

    rC = np.loadtxt(event.artist.get_picker() + '/rC.txt', comments='#', delimiter=',')
    agrain = np.loadtxt(event.artist.get_picker() + '/agrain.txt', comments='#', delimiter=',')
    sigma = np.loadtxt(event.artist.get_picker() + '/sigma.txt', comments='#', delimiter=',')
    Frag = np.loadtxt(event.artist.get_picker() + '/Frag.txt', comments='#', delimiter=',')
    Drift = np.loadtxt(event.artist.get_picker() + '/Drift.txt', comments='#', delimiter=',')

    plot00 = ax2.pcolormesh(rC,agrain,sigma, vmin=-10, vmax=1)
    ax2.plot(rC, Frag, label='Fragmentation barrier')
    ax2.plot(rC, Drift, label='Drift  barrier')
    ax2.set_xscale('linear')
    ax2.set_yscale('log')
    ax2.set_ylim(bottom=agrain[0])
    plt.colorbar(plot00, ax=ax2)
    ax2.set_xlabel('Radius[AU]')
    ax2.set_ylabel('Particle mass [g]')
    ax2.legend()
    plt.draw()
    plt.show()

def callback(event):

    """The callback for updating the figure when the buttons are clicked"""

    global Names_states
    global allSim_states
    global op_selected
    global Nsim
    global Mp1_selected
    global Mp2_selected
    global Mp3_selected
    global rp1_selected
    global rp2_selected
    global rp3_selected
    global alpha_selected
    global gamma_selected
    global Mgas_selected
    global tp2_selected
    global tp3_selected
    global Texp_selected
    global TR0_selected
    global t_selected
    global legend_states
    global video_states
    global vf_selected
    global docu_states
    global docu_Selected
    """global example_states"""
    """gobal_example_selected"""


    """Screen buttons"""

    docu_states =docu_buttons.get_status()
    op_states = op_buttons.get_status()
    allSim_states = allSim_buttons.get_status()
    Names_states =Names_buttons.get_status()
    legend_states =legend_buttons.get_status()
    video_states = video_buttons.get_status()

    """Parameter buttons"""

    alpha_states = alpha_buttons.get_status()
    gamma_states = gamma_buttons.get_status()
    tp3_states = tp3_buttons.get_status()
    tp2_states = tp2_buttons.get_status()
    Mgas_states = Mgas_buttons.get_status()
    t_states = t_buttons.get_status()
    Texp_states = Texp_buttons.get_status()
    Mp1_states = Mp1_buttons.get_status()
    Mp2_states = Mp2_buttons.get_status()
    Mp3_states = Mp3_buttons.get_status()
    vf_states = vf_buttons.get_status()
    """example_states = example_buttons.get_status()"""

    """Deactivated buttons"""

  # TR0_states = TR0_buttons.get_status()
  # rp1_states = rp1_buttons.get_status()
  # rp2_states = rp2_buttons.get_status()
  # rp3_states = rp3_buttons.get_status()

    """"List of selected parameter buttons"""
    alpha_selected = list(compress(alpha_values, alpha_states))
    gamma_selected = list(compress(gamma_values, gamma_states))
    tp3_selected = list(compress(tp3_values, tp3_states))
    tp2_selected = list(compress(tp2_values,tp2_states))
    Mgas_selected = list(compress(Mgas_values, Mgas_states))
    Texp_selected = list(compress(Texp_values, Texp_states))
    Mp1_selected = list(compress(Mp1_values, Mp1_states))
    Mp2_selected = list(compress(Mp2_values, Mp2_states))
    Mp3_selected = list(compress(Mp3_values, Mp3_states))
    vf_selected =  list(compress(vf_values, vf_states))
    # rp1_selected = list(compress(rp1_values, rp1_states))
    # rp2_selected = list(compress(rp2_values, rp2_states))
    # rp3_selected = list(compress(rp3_values, rp3_states))

    t_selected = list(compress(t_values, t_states))
    op_selected  =list(compress(op_values, op_states))

    """example_selected =list(compress(op_values, op_states))"""

    conditions(event)

def Docu(event):

    """ When calling it, it plots a text in an aditional figure explaining the meaning of some of the variables used"""

    figT = plt.figure(figsize=(9,6))
    box= dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    figT.text(0.05, 0.15,' \n \n Interface for the visualization and comparison between the instensity flux derived from simulations using \n \n dustpy. The profiles  measured by Andrews, 2018 and Isella et al, 2016 used as reference.  \n  \n  By clicking o a line, a countor plot of the simulation and its corresponding parameters appear.\n \n  \n \n ' r'$\alpha$'':  Viscosity parameter \n\n $v_\mathrm{frag}$:   Fragmentation velocity of the dust grains [cm/s]. \n \n $\gamma$:       Exponential factor for the unperturbed density.  \n \n $T_{exp}$:   Exponential factor for the temperature profile. \n \n $TR_{0}$:  Cut-off radius for the temperature profile [Au].\n\n $M_\mathrm{g}$: Mass of the gas in the disk as a ratio of the Star mass. \n\n $r_c$: Cut-off radius for the unperturbed gas density profile [Au]. \n \n $T_{p3}$:    Time when rthe third planet starts growing, in comparison to the  of the simulation [Myr].'\
    ,fontsize=10,bbox = box)
    figT.show()


"""Buttons panel control"""
alpha_buttons.on_clicked(callback)
gamma_buttons.on_clicked(callback)
Mgas_buttons.on_clicked(callback)
Texp_buttons.on_clicked(callback)
Mp1_buttons.on_clicked(callback)
Mp2_buttons.on_clicked(callback)
Mp3_buttons.on_clicked(callback)
vf_buttons.on_clicked(callback)
tp2_buttons.on_clicked(callback)
tp3_buttons.on_clicked(callback)
t_buttons.on_clicked(callback)
op_buttons.on_clicked(callback)
"""example_buttons.on_clicked(callback)"""
# rp1_buttons.on_clicked(callback)
# rp3_buttons.on_clicked(callback)
# rp3_buttons.on_clicked(callback)

"""Buttons main screen"""
legend_buttons.on_clicked(callback)
refresh_buttons.on_clicked(callback)
allSim_buttons.on_clicked(callback)
video_buttons.on_clicked(callback)
Names_buttons.on_clicked(callback)
docu_buttons.on_clicked(callback)

"""Show Plot """
plt.show()
