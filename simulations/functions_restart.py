import numpy as np
import os
import glob
import h5py

from dustpy.sim.utils import bindFunction

def loadSimulation(sim, dir=None, files="data*.hdf5"):
	#This function restarts the simulation from the written file in the simulation output directory by default.
	#It will write the following snapshots in a new directory with the suffix "_restart".
	#You can copy them back into the initial directory manually, but be careful of not overwriting the initial condition file.
	if dir == None:
		dir = sim.pars.outputDir
	restart_file = readFilesFromDir(dir,files)[-1]

	sim.t=ReadTime(restart_file)
	bindFunction(
	    sim,
	    "initialGasSurfaceDensity",
	    ReadGasDensity,
	    filename=restart_file
	)
	bindFunction(
	    sim,
	    "initialTemperatureProfile",
	    ReadGasTemperature,
	    filename=restart_file
	)
	bindFunction(
	    sim,
	    "initialDustSurfaceDensity",
	    ReadDustDensity,
	    filename=restart_file
	)
	sim.pars.outputDir +="_restart"
	print("Restarting simulation from "+restart_file)




def ReadGasDensity(sim, r, filename):
	return readFieldFromFile("gas/Sigma",filename)

def ReadGasTemperature(sim, r, filename):
	return readFieldFromFile("gas/T",filename)


def ReadDustDensity(sim, filename):
	return readFieldFromFile("dust/Sigma",filename)

def ReadTime(filename):
	return readFieldFromFile("t",filename)


#####################
#Functions from Plotting.py

def readFilesFromDir(dir, files):
    if os.path.isdir(dir):
        files = glob.glob(os.path.join(dir, files))
    elif os.path.isfile(dir):
        files = [dir]
    elif type(dir) is list:
        files = dir
    else:
        raise TypeError('dir needs to be a string or list of strings')

    files.sort()
    return files


def readFieldFromFile(field, file):
    f = h5py.File(file, "r")
    val = f[field].value
    return val 
