
# some of the basic libraries that can be useful
import astropy.constants as cc
import os
import dustpy
from dustpy.plotting.plot import readFilesFromDir, readFieldFromFile
import h5py
from helper import planck_B_nu, convolve, get_dsharp_data
from scipy.interpolate import interp2d
import dsharp_opac as op
import dustpy
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy

"File containing some of the functions needed to set up  the simulations "

class c:

    """a class with some useful constants , in cgs units"""

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

au =c.au


def Pressure(name,num_of_file):

    " Gives an array of pressure values at each radial grid coordinate, for a given temperature and value of the gas surface density"

    with h5py.File(str(name) + '/data00' + str(num_of_file) + '.hdf5') as f:
        #t = f['dt'][()]
        sig_g = f['gas/Sigma'][()]
        sig_d = f['dust/Sigma'][()]
        r_sim = f['grid/r'][()]  # radial grid --> 100
        m = f['grid/m'][()]
        temp = f['gas/T'][()]

    pressure = np.sqrt(temp)*omega(r_sim)*sig_g

    return r_sim, pressure

def SurfDens(direct, num_of_file):


    """function useful for plotting the gas and dust surface density, contains:

    [0] Gas surface density
    [1] Dust surface density
    [2] Its corresponding radial coordiante in AU

    ----

    Arguments :

    direct : directory were the simulation is stored
    num_of_file : snapshot for which the contour plot will be done

    ---

    """

    with h5py.File(str(direct) + '/data00' + str(num_of_file) + '.hdf5') as f:
        #t = f['dt'][()]
        sig_g = f['gas/Sigma'][()]
        sig_d = f['dust/Sigma'][()]
        r_sim = f['grid/r'][()]  # radial grid --> 100

    return r_sim / c.AU,sig_g, sig_d.sum(axis=1),

def ContourPlot(name,numsim):


    """function useful for making a contour plot with its corresponding fragmentation and
    drift barrier. All the legend, colormash and frag barrier compution is done in the routine

    , contains:

    [0] Gas surface density
    [1] Dust surface density
    [2] Its corresponding radial coordiante in AU
    """


    with h5py.File(str(name) + '/data00'+str(numsim) + '.hdf5') as fC:
     rC = fC['grid/r'][()]
     mC = fC['grid/m'][()]
     sig_dC = fC['dust/Sigma'][()]
     csC = fC['gas/cs'][()]
     alphaC = fC['gas/alpha'][()]
     VfragC = fC['dust/vFrag'][()]
     SigmaC = fC["gas/Sigma"][()]
     omegaC = fC["grid/OmegaK"][()]
     St = fC["dust/St"][()]
     d2g = fC["dust/dust2gasRatio"][()]
     rIntC = fC["grid/rInt"][()]
     rhoC = fC['dust/rhoBulk'][()][0,0]
     agrain = fC['dust/a'][()][0]
     Vk = omegaC * rC

    FragBarrier = VfragC*VfragC / csC/csC/alphaC /3
    FragBarrier *= 2 * SigmaC / np.pi /rhoC
    p = SigmaC * omegaC * csC
    _f = scipy.interpolate.interp1d(np.log10(rC/c.AU), np.log10(p), fill_value='extrapolate')
    pInt = 10.**_f(np.log10(rIntC/c.AU))
    Diff_pint = np.diff(pInt)
    Diff_rint = np.diff(rIntC/c.AU)
    gammaC = np.abs(rC / p * Diff_pint / Diff_rint/c.AU)
    DriftBarrier = 2 * d2g * SigmaC * Vk*Vk /csC/csC/gammaC/ np.pi /rhoC
    fig2 = plt.figure()
    ax2 = fig2.add_axes([0.1, 0.1, 0.5, 0.5])

    plot00 = ax2.pcolormesh(rC/c.AU,agrain,np.log10(sig_dC).T, vmin=-10, vmax=1)
    ax2.plot(rC/c.AU, FragBarrier, label='Fragmentation barrier')
    ax2.plot(rC/c.AU, DriftBarrier, label='Drift barrier')
    ax2.set_xscale('linear')
    ax2.set_yscale('log')
    ax2.set_ylim(bottom=agrain[0])
    plt.colorbar(plot00, ax=ax2)
    ax2.set_xlabel('Radius[AU]')
    ax2.set_ylabel('Particle mass [g]')
    # ax2.legend()
    plt.draw()
    plt.show()

class Extra_var(object):


    """ this class allows for storage of extra variables that do not directly appear in dustpy , but which may be useful when  performing a simulation with a given planet mass or
     other propierties / variables . One can delete the variables that will not be used, or define new ones if required."""


    def __init__(self, name,Mp1=None,Mp2=None,Mp3 = None,rp1=None,rp2=None,rp3=None,alpha =None,gamma = None,Mgas = None,rc=None ,tp2 =None,tp3 = None,l3 =None, profile = None):
        self.name = name
        self.Mp1 = Mp1
        self.Mp2 = Mp2
        self.Mp3 = Mp3
        self.rp1 = rp1
        self.rp2 = rp2
        self.rp3 = rp3
        self.alpha = alpha
        self.gamma = gamma
        self.Mgas  = Mgas
        self.rc = rc
        self.tp2 = tp2
        self.tp3 =  tp3
        self.l3 = l3
        self.profile = profile
    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


def save_object(obj, filename):
    pickle_out = open(filename, 'wb')
    pickle.dump(obj, pickle_out)
    pickle_out.close()


import cv2

def Video(output):

    """creates a video  of the contour plot of a certain simulation using all of its snapshots.

    Arguments:

    output : simulation path

    """

    cap =cv2.VideoCapture(output)
    while(cap.isOpened()):
        ret, frame = cap.read()

        cv2.COLOR_BGR2HSV
        cv2.imshow('frame',frame)
        cv2.waitKey(60)

    cap.release()

def Chi2(r,X,sigma,profile='isella.txt'):

    """  given an intensity flux obtained from an observation, it compares it with the desired observational intensity flux and returns its chi2 value

    Arguments :

    r : radial coordinate of the  intensity flux resulting from the simulation, units in arcseconds
    X : Simulated intensity flux
    profile : profile corresponding to the observation for which the comparison will be done. In column 1 the radial  coordinate, in column 2
    the intensity flux and in column 3 its corresponding uncertainty, delimiter = ','

    Returns :

    Value of Chi2

    """

    R,I,sigma = np.loadtxt(profile,comments='#',delimiter=',')

    return np.sum(((I-X)/sigma)**2)

def fitting(names_prime,s,profile = 'isella.txt'):

    """
    given a list of names as the different paths to the simulations, it looks for the 'sth' simulations that  have a lower Chi2 in comparison
    to the chosen profile or observation, and also at which time t that happens.

    Arguments :

    names_prime : paths to the different simulations. If all the folders in the directory correspond to a simulation, the following input is recomended:

    names_prime = np.asarray([str(f) for f in glob.glob('*')])

    s : the number of outcomes in form of simualtion names desired. Eg: if s = 10, then it gives you the 10th simulations that match better the profile"

    profile : keyword argument. To be set to the desired intensity flux to be compared. COlumn1 : R , column2: profile, column 3: sigma. delimiter =','

    Returns:

    s number of plots, each one appears after the previous one is closed. 0 corresponds to the best and 9 to the 'least' bestself.

    """


    names =[]
    Min=[]
    time=[]

    for i in names_prime:


      if os.path.isdir(str(i)) ==True: # " single files are discarted"
            names.append(str(i))

    for i in names:

        # print(i)
        test =[]
        for j in range(0,51): # range of snapshots for which the Chi2 is calculated


          try:

           Int = dens_to_int(str(i),j)
           I_sim = Int[3]
           r = Int[0]

           chisquare = Chi2(I_sim,r,profile = profile)
           test.append(chisquare)

          except:

            print('no data'+str(j)+'for' + i)

        if len(test)>0:

            # print(test)
            #
            # print(min(test))

            Min.append(min(test))

            time.append(test.index(min(test)))

    best = np.min(Min)  # the minimum CHI2 among all the simulations, if the following lines are uncommented, one obtains the name of the best sim and at which time Chi2 is minimized
    # print(names[Min.index(best)])
    # print(time[Min.index(best)])
    list_best = sorted(Min)
    for i in range(0,s):

        print(str(i)+':',names[Min.index(list_best[i])],time[Min.index(best)])
        I = dens_to_int(str(names[Min.index(list_best[i])]),time[Min.index(best)])
        plt.semilogy(I[0],I[3],label = 'profile' + str(i))
        plt.semilogy(I[0],I[1],label =  str(profile))
        plt.legend()
        plt.ylim(1e-16,1e-11)
        plt.xlim(0,3)
        plt.show()

from dustpy.sim.utils import readFromDump

def CompareVars(A,names,all = True):

    """"Checks differencess between  simulation A and all the simulations in 'names'

    Arguments:

    A : main simulation

    names = array of simulations whose parametes will be compared with A

    Prints :

    The parameters for which A and the rest of the simulations differ. The parameters used are the ones appearing in Extra vars. It will
    work if the simulations have been perforemd using the jupyter notebook in this folder.

    """

    with open(str(A)+'/Extra_vars.pickle', 'rb') as F:

         d = pickle.load(F)

    D = readFromDump(str(A)+'/dustpy.dmp')

    for i in names:

         if os.path.isdir(str(i))==True:

              with open(str(i)+'/Extra_vars.pickle', 'rb') as G:
                     d1 = pickle.load(G)

              print('Diff between '+str(A)+' and ' + str(i)+': \n')
              print(d.__dict__.items()^d1.__dict__.items())

         if all==True:

              D = readFromDump(str(i)+'/dustpy.dmp')

              print('\n')

              print(d.__dict__.items()^d1.__dict__.items())

              print(' --------------- \n ')

def ContourPlot(name,numsim):


    """function useful for making a contour plot with its corresponding fragmentation and
    drift barrier. All the legend, colormash and frag barrier compution is done in the routine

    , contains:

    [0] Gas surface density
    [1] Dust surface density
    [2] Its corresponding radial coordiante in AU
    """


    with h5py.File(str(name) + '/data00'+str(numsim) + '.hdf5') as fC:
     rC = fC['grid/r'][()]
     mC = fC['grid/m'][()]
     sig_dC = fC['dust/Sigma'][()]
     csC = fC['gas/cs'][()]
     alphaC = fC['gas/alpha'][()]
     VfragC = fC['dust/vFrag'][()]
     SigmaC = fC["gas/Sigma"][()]
     omegaC = fC["grid/OmegaK"][()]
     St = fC["dust/St"][()]
     d2g = fC["dust/dust2gasRatio"][()]
     rIntC = fC["grid/rInt"][()]
     rhoC = fC['dust/rhoBulk'][()][0,0]
     agrain = fC['dust/a'][()][0]
     Vk = omegaC * rC

    FragBarrier = VfragC*VfragC / csC/csC/alphaC /3
    FragBarrier *= 2 * SigmaC / np.pi /rhoC
    p = SigmaC * omegaC * csC
    _f = scipy.interpolate.interp1d(np.log10(rC/c.AU), np.log10(p), fill_value='extrapolate')
    pInt = 10.**_f(np.log10(rIntC/c.AU))
    Diff_pint = np.diff(pInt)
    Diff_rint = np.diff(rIntC/c.AU)
    gammaC = np.abs(rC / p * Diff_pint / Diff_rint/c.AU)
    DriftBarrier = 2 * d2g * SigmaC * Vk*Vk /csC/csC/gammaC/ np.pi /rhoC
    fig2 = plt.figure()
    ax2 = fig2.add_axes([0.1, 0.1, 0.5, 0.5])

    plot00 = ax2.pcolormesh(rC/c.AU,agrain,np.log10(sig_dC).T, vmin=-10, vmax=1)
    ax2.plot(rC/c.AU, FragBarrier, label='Fragmentation barrier')
    ax2.plot(rC/c.AU, DriftBarrier, label='Drift barrier')
    ax2.set_xscale('linear')
    ax2.set_yscale('log')
    ax2.set_ylim(bottom=agrain[0])
    plt.colorbar(plot00, ax=ax2)
    ax2.set_xlabel('Radius[AU]')
    ax2.set_ylabel('Particle mass [g]')
    # ax2.legend()
    plt.draw()
    plt.show()


def PrintVars(names):

    """ If some of the variables  in the simulation have been stored in  a file Extra_vars.pickle and others in the original dmp file , with this function one quickly checks some of those variables at the same times. Here

    you can delete the ones that are being showed now and re-write the desired ones EG : one could keep just the masses  and dust to gas ratio if the aim is to compare planet masses values amog the different

    simulations.

    Arguments:

    names : array of simulations for which the parameters will be shown.  If one wants all the simulations to be compared : names = np.asarray([str(f) for f in glob.glob('*')])

    """

    for i in names:

     if os.path.isdir(str(i)) ==True:
        with open(str(i)+'/Extra_vars.pickle', 'rb') as F:
             d = pickle.load(F)
        D = readFromDump(str(i)+'/dustpy.dmp')
        print('Name', 'Mp1' , 'Mp2'  ,  'Mp3 ' ,' tp2'  ,'tp3 ',  'Mgas' ,  'gamma' ,  'alpha' ,'time' ,'vFrag','rc','Texp','dust2gasRatio')

        print(str(i),"{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.3f},{:.2f},{:.4f},{:.2f},{:.2f},{:.2f},{:.2f}".format(d.Mp1,d.Mp2,d.Mp3,d.tp2/1e6,d.tp3/1e6,d.Mgas,d.gamma,d.alpha\
        ,D.t/c.yr/10**6,D.dust.vFrag[0],d.rc,D.ini.gas.TExp,D.ini.dust.dust2gasRatio))


"""Isella Profile"""

def gaussian(theta,theta_n, a_n, b_n):

    return a_n * \
    np.exp(-((theta - theta_n) / b_n)**2)


def U_ise(theta):

  a, b = 0.99, 0.76
  a1, b1, theta1 = 0.55, 0.23, 0.44
  a2, b2, theta2 = 0.17, 0.15, 0.81
  a3, b3, theta3 = 0.06, 0.28, 1.13

  U_a1,U_theta1,U_b1 = 0.02,0.01,0.01
  U_a2,U_theta2,U_b2 = 0.01,0.01,0.01
  U_a3,U_theta3,U_b3 = 0.01,0.01,0.01

  def  U_g(theta):

      U_a = 0.01
      U_b = 0.02

      return np.exp(-(theta/b)**2)*np.sqrt( U_a**2 + 4*(a*theta**2/b**3)**2*U_b**2)

  def U_gn(theta,thetan,an,bn,U_thetan,U_an,U_bn):

      return np.exp(-((theta-thetan)/bn)**2)*np.sqrt(U_an**2 + 4*(an*(theta-thetan)**2/bn**3)**2*U_bn**2 +\
      4*an**2/bn**4*(theta-thetan)**2*U_thetan**2)

  return np.sqrt(U_g(theta)**2 + U_gn(theta,theta1,a1,b1,U_theta1,U_a1,U_b1)**2+U_gn(theta,theta2,a2,b2,U_theta2,U_a2,U_b2)**2
    +U_gn(theta,theta3,a3,b3,U_theta3,U_a3,U_b3)**2)

def brightness(theta):

    a, b = 0.99, 0.76
    a1, b1, theta1 = 0.55, 0.23, 0.44
    a2, b2, theta2 = 0.17, 0.15, 0.81
    a3, b3, theta3 = 0.06, 0.28, 1.13

    return gaussian(theta,0, a, b) - gaussian(theta,theta1, a1, b1) - gaussian(theta,theta2, a2, b2) - gaussian(theta,theta3, a3, b3)
