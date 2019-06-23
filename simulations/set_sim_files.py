import glob
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import dustpy
from dens_v2 import dens_to_int_v2
import pickle


"Function for the setting of the intensity for all of the available simulations, using two different opacities."

names_prime = np.asarray([str(f) for f in glob.glob('post*')])

Int=[]
R=[]
Ise=[]

NO_sim = ['videos','__pycache__','opacities']


for i in names_prime:

    if os.path.isdir(str(i)) ==True:

        Ise2 = dens_to_int_v2(str(i),50,opacities = 'default_opacities_smooth.npz')
        Int = Ise2[2]
        R   = Ise2[0]
        np.savetxt(str(i)+'/I_R_v2',(R,Int),delimiter=',', header="I,R",fmt='%.18e')

        Ise1 = dens_to_int_v2(str(i),50,opacities = 'default_opacities.npz')
        Int = Ise1[2]
        R   = Ise1[0]
        np.savetxt(str(i)+'/I_R_v1',(R,Int),delimiter=',', header="I,R",fmt='%.18e')
