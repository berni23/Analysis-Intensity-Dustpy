

Interface folder, contains:


1 . bestFit.ipynb: Routine for parallel running simulations using dustpy and the kanagawa profile.
------------------------------------------------------------------
------------------------------------------------------------------

2 . dens_v2.py: Contains dens_to_int_v2. A function for the conversion from dust surface density as an outcome of a dustpy simulation, to 
the corresponding intensity flux one would observe, given a distance to the system , the wavelength and the opacities.

------------------------------------------------------------------
------------------------------------------------------------------

3 . sim_functions.py: Contains  some the functions necessary to perform the simulations, as well as analyzing them.

------------------------------------------------------------------
------------------------------------------------------------------

4 . helper.py: Contains some pre-defined functions line the planck distribution or  a convolution function, used in dens_v2.

------------------------------------------------------------------
------------------------------------------------------------------

5 . kanagawa.py: Contains the necessary functions to implement the kanagawa profile as the gas surface density function in a dustpy simulation.

------------------------------------------------------------------
------------------------------------------------------------------

6 . set_sim_files.py : File with a routine for re-calculating the intensity fluxes for a set of simulations and storing into a file I_R_v1 and I_R_v2 for two different opacities. 
Those files are used in the interface for plotting the data.

------------------------------------------------------------------
------------------------------------------------------------------

7 .functions_restart.py : Contains several functions used in order to  restart a given simualtion. That is done using the routine in LoadSim.ipynb .

------------------------------------------------------------------
------------------------------------------------------------------

8 . LoadSim.py :  Routine for restarting simulations in dustpy and  running them in parallel.

------------------------------------------------------------------
------------------------------------------------------------------
9.  bestFit.py : ".py" version of bestFit.ipynb notebook. To be used in gustl. It can be created by tipying "jupyter nbconvert --to python bestFit.ipynb"
on the terminal. To choose the number of cores : export OP_NUM_THREADS== n° of cores.

-----------------------------------------------------------------
------------------------------------------------------------------

10. Data_evolving : Simulation for the gas surface density of the HD 163296 protoplanetary system azimuthally averaged.

------------------------------------------------------------------
------------------------------------------------------------------

11. Opacities : Folder with three different opacities for a given wavelenght and grain size

------------------------------------------------------------------
------------------------------------------------------------------
