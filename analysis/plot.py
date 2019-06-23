import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
import dustpy
from dustpy.sim.utils import bindFunction,readFromDump
from scipy  import interpolate
import glob

import sys

sys.path.insert(0,'../simulations')

from dens_v2 import dens_to_int_v2

import matplotlib.pyplot as plt

g = input("Enter the name pattern of simulations you want to plot : ")

print(g)

files = np.asarray([str(f) for f in glob.glob('../simulations/'+str(g))])
names =[]
names_sim =[]
I =[]

Ise =np.loadtxt('isella.txt')

k=0
for i,j in enumerate(files):

 if os.path.isdir(str(j)) == True:

     names.append(files[i])


for i in names:

     try:

        Inte =dens_to_int_v2(i,50)
        I.append(Inte[0])
        I.append(Inte[2])
        names_sim.append(i)
        k+=1

     except:

        print('No snapshot 50 for '+str(i))

for j in range(0,k):

    i = names_sim[j][15:-1]+names_sim[j][-1]

    plt.semilogy(I[j*2],I[j*2+1],label = str(i))
plt.semilogy(Ise[0],Ise[1],label ='Isella Profile')
plt.ylabel(r'$I_{\nu}$ $[erg/cm^2/Hz/Sr]$', fontsize=8)
plt.xlabel(r'$\theta$ [arcsec]', fontsize=8)
plt.ylim(1e-20,1e-11)
plt.xlim(0,3)

plt.legend()
plt.show()
