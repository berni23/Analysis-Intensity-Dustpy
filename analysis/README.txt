 
analysis folder, contains:

------------------------------------------------------------------
------------------------------------------------------------------

1 - interface.py : Interface enabling for a visualization of the  convolved intensity data obtained from the simulations done with dustpy. 

When compiling, a screen for the visualization of the data and  a panel control appear. In the right hand side of the plot,

we have seven different options. From top to bottom :

Show all : An option to make all the simulations available appear into the plot. That gives us an idea of how of the total the parameter space has been covered and which part is yet to be explored.

Refresh : An option that actualizes the content.

Legend : By clicking this option, the legend of the simulations plotted in the screen appear. 

Show video :  By selecting one simulation and then clicking this button, a movie of the evolution regarding the radial distribution of the grain mass as a function of time is showed.

Print simulations : This command prints the names of the simulations appearing in the screen at the given time. This option is useful whenever we constraint

the simulations into a certain parameter range and we want them listed for further study.If we proceed to constraint even more our parameter space, we will be able to again

print the simulation names and compare how many are left. It also gives the number of parameter permutations that have not been explored yet.

Parameter documentation: By clicking this option, a breve explanation for each parameter subject to study and its physical meaning appears.

------------------------------------------------------------------
------------------------------------------------------------------

2 - Isella.txt and HD_profile.txt : two of the profiles used in my case for a comparison in the interface.py. To be changed for the desired ones.

------------------------------------------------------------------
------------------------------------------------------------------

3 - plot.py : A py file containing a plotting routine for the intensity flux derived from the simulations.

------------------------------------------------------------------
------------------------------------------------------------------

4- Analysis gas profile. In contrast with the interface, in this case we take a look on the gas profile itself, and see how could we improve it to assimilate more

to the hydrodinamic simulation. The directory contains a folder with an MC routine in order to find the parameters that better assimilate to the hydro simulation,

and also another folder with a widget to be used to check which function would be better to characterize the parameter q (Mplanet/Mstar) as a function of time.


------------------------------------------------------------------
------------------------------------------------------------------
