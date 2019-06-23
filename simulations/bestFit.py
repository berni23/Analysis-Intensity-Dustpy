#!/usr/bin/env python
# coding: utf-8

# # An executable notebook
# This notebook can also be converted to a script with 
# 
#     jupyter nbconvert --to script multiproc.ipynb
#     
# and then executed with
# 
#     python multiproc.py [ARGS]
#     
# If ipython magic is used, it need to be called not with `python` but with `ipython`, but then arguments cause issues.

# In[1]:


import argparse
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="No more knots can be added because the additional knot would coincide with an old one. Probable cause: s too small or too large")
warnings.filterwarnings("ignore", message="numpy.dtype size changed, may indicate binary incompatibility. Expected 88 from C header, got 96 from PyObject")

from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
import shutil
import pickle
import os
import sys
import dustpy
#import pandas as pd
from dustpy.sim.utils import bindFunction
from scipy  import interpolate
#from widget import plotter
# to find out if we are running interactively
import __main__
is_interactive = not hasattr(__main__, '__file__')


# In[2]:


parser = argparse.ArgumentParser(description='This python notebook is executable',formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('-o','--opacity', help='what opacity to use',   type=str, default='ricci', choices=['ricci','dsharp'])
parser.add_argument('-n','--number',  help='which model to run',    type=int, default=-1)
parser.add_argument('-c','--cpus',    help='how many cores to use', type=int, default=4)

parser.description ='Here we can put more description'
parser.description+='And even more if necessary'
# use parser default if interactive
if not is_interactive:
    print('not interactive, will parse arguments')
    argsin = parser.parse_args()
else:
    print('running interactively, will use some defaults')
    argsin = parser.parse_args(['-o','dsharp'])
    
print('we are using the opacity from {}'.format(argsin.opacity))


# Define a worker function that takes some time to run

# In[3]:


import matplotlib.pyplot as plt
from importlib import reload
import kanagawa
from dens_v2 import dens_to_int_v2
reload(kanagawa)
globals().update({k:v for k,v in kanagawa.__dict__.items() if not k.startswith('_')}) 
from scipy.interpolate import interp1d 
from dustpy import plotting
import datetime
import numpy as np 
now = datetime.datetime.now()
import sim_functions
from sim_functions import save_object,Extra_var

def worker_function(tp2,tp3,numSim):
    
    """ Function with all the necessary features to perform a simulation in dustpy using the kanagawa profile. To change the function argumnents for the desired ones."""

    A = dustpy.sim.Simulation()
    
    """ Naming of the simulation : post_+ date where the simulation is initialized + simulation number  ( if we are performing several at once)"""
    
    A.pars.outputDir = 'post_' + str(now.day) + '_'+ str(now.month)+ '_'+str(numSim)
    
    """ Parameters to change an existing in dmp file"""
    
    A.ini.dust.allowDriftLimitedParticles = True
    A.pars.gasAdvection =  False
    A.allowDriftlimit   =  True
    A.pars.excludeAttr  = ['dust/jac','dust/cFrag','dust/cStick','dust/kFrag','dust/kStick','dust/vRel']
    A.ini.dust.vFrag = 2e3
    A.ini.star.M =2.3*c.Msun
    A.ini.gas.T0 =24
    A.ini.gas.TR0=100*c.AU
    A.ini.gas.TExp=0.6
    A.verbose =4
    A.ini.dust.dust2gasRatio = 1.5e-2
    A.snapshots = np.linspace(2,5*1e6,50)*c.yr
    #A.snapshots = np.logspace(2,6.6,50)*c.yr
    A.grid.rInt = np.linspace(2,500,301)*c.AU 
    ALPHA = 1e-3
    A.gas.alpha =ALPHA
     
    """Parameters that are not considered  in dustpy by default, but included in the gas surface density profile"""
    
    M_p1=0.6
    M_p2= 1
    M_p3 = 1.3
    t_p2 = tp2 #1
    t_p3 = tp3
    M_gas = 0.2
    g_amma = 0.2
    r_p1 = 56
    r_p2 = 83
    r_p3 = 125
    r_c = 100
    
    """Initialization gas surface density"""
    
    def initialGas(A,r):
        
        return kanagawa_time_3gap(A,r,0,M_gas)
        
    def updateGas(A):
        A.gas.Sigma = kanagawa_time_3gap(A,A.grid.r, A.t,M_gas,Mp1=M_p1,Mp2=M_p2,Mp3 = M_p3,tp2 = t_p2,tp3 = t_p3,gamma = g_amma,rp1 =r_p1 ,rp2 =r_p2,rp3 = r_p3,rc = r_c,alpha = ALPHA)
        
    def updateAlpha(A):
        return np.array([ALPHA for i in range(0,len(A.grid.r))])
     
    bindFunction(A,'initialGasSurfaceDensity',initialGas)
    bindFunction(A,'gasSystole', updateGas)
    bindFunction(A,'alpha',updateAlpha)

    """ Initialization of the simulation"""
    
    print('Running dustpy simulation n'+str(numSim)+'..' )
    
    A.initialize()
    
    """ Storage of extra variables we are interested at but that are not stored in the simulation by default"""
    
    Variables  = Extra_var(A.pars.outputDir,Mp1=M_p1,Mp2=M_p2,Mp3 = M_p3,alpha =ALPHA,tp2 = t_p2*1e6,tp3 = t_p3*1e6,Mgas = M_gas,gamma = g_amma,rp1 =r_p1 ,rp2 =r_p2, rp3 = r_p3,rc = r_c,profile ='hydro')
    save_object(Variables,str(A.pars.outputDir) +'/Extra_vars.pickle')
     
    """Evolution of the simulation"""  
    
    A.evolve()
    
    """ Getting the variables needed for a contour plot"""                           
    with h5py.File(str(A.pars.outputDir) + '/data0050.hdf5') as fC: 
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
    
    """ The intensity derived from the dust surface denstity at the last snapshot is stored"""
    
    Ise1 = dens_to_int_v2(A.pars.outputDir,50,opacities = 'default_opacities.npz')
    
    R=Ise1[0]
    Int=Ise1[2] 
    
    np.savetxt(A.pars.outputDir+'/I_R_v1',(R,Int),delimiter=',', header="I,R",fmt='%1.8f')
    
    Ise2 = dens_to_int_v2(A.pars.outputDir,50,opacities = 'default_opacities_smooth.npz')
    
    R = Ise2[0]
    Int = Ise2[2] 
    
    np.savetxt(A.pars.outputDir+'/I_R_v2',(R,Int),delimiter=',', header="I,R",fmt='%1.8f')
    
    """ The variables neded to create a contour plot are also stored at each simulation folder, to see the contourplot just
    use the ContourPlot function in sim_funcions.py file"""

    _f = interp1d(np.log10(rC), np.log10(p), fill_value='extrapolate') 
    pInt = 10.**_f(np.log10(rIntC)) 
    Diff_pint = np.diff(pInt) 
    Diff_rint = np.diff(rIntC) 
    gammaC = np.abs(rC / p * Diff_pint / Diff_rint) 
    DriftBarrier = 2 * d2g * SigmaC * Vk[0]*Vk[0] /csC/csC/gammaC/ np.pi /rhoC 
    np.savetxt(str(A.pars.outputDir)+'/rC.txt',rC,delimiter = ',')
    np.savetxt(str(A.pars.outputDir)+'/agrain.txt',agrain,delimiter = ',')
    np.savetxt(str(A.pars.outputDir)+'/sigma.txt' ,np.log10(sig_dC).T,delimiter = ',')
    np.savetxt(str(A.pars.outputDir)+'/Frag.txt'  ,FragBarrier,delimiter = ',')
    np.savetxt(str(A.pars.outputDir)+'/Drift.txt' ,DriftBarrier,delimiter = ',')
      
    """ Movie of the Contour plot (population size vs position in arcsec)"""                           
    #plotting.movie(str(A.pars.outputDir))


# In[ ]:


Args =([0.5,0.5,1],[1,1,2],[1.5,1.5,3],[2,2,4])

with Pool(argsin.cpus) as p:

          p.starmap(worker_function,Args)


# In[ ]:


""" Simulation with the desired variables. The ones we want to change have to be an argument of the worker_funcion. The last argument of worker_function should always be the number
of the simulation.

3 options:

1- Just running one sim:

        worker_function(a,b,..,1)

2 - Several simulations:

        Args =([1.2,0.8,0.9,1],[0.8,0.8,0.9,2],[1,1,0.9,3])

        with Pool(argsin.cpus) as p:

          p.starmap(worker_function,Args)

3 - Several simulations by looping through all the desired parameters: In this case all the combinations for tp2 and tp3 : 9 simulations

        tp2 = [0,0.5,1]
        tp3 = [1,2,3] 
        Args = []
        k =0 
        for i in tp2:
          for j in tp3:
            k = k+1
            Args.append([i,j,k])

        with Pool(argsin.cpus) as p:

          p.starmap(worker_function,Args)"""


# In[ ]:


""" Commands for the conversion for this file to a py file, to  run it in gustl
jupyter nbconvert --to python bestFit.ipynb

export OP_NUM_THREADS== number of cores """


# In[ ]:


""" A final message to know the simulations are done"""

print("All simulations complete!! ... it took long, but it was worth it!!")

