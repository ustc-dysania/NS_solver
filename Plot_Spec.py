from matplotlib import pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

gravitational_constant = 6.6743e-11  # m^3 kg^-1 s^-2
speed_of_light = 299792458.0         # m/s
solar_mass = 1.9884099021470415e+30  # Kg

#? get love number to 10^36 g cm^2 s^2
unitconv      = gravitational_constant**4/speed_of_light**10*1e3*1e4         #? convert kg^5 to g cm^2 s^2

width_NICER = 14.255-11.959
height_NICER = 1.594-1.299
NICER_Miller = Rectangle((11.959,1.299), width_NICER, height_NICER, facecolor='violet', fill=True, alpha=0.5)

NICER_Riley = Rectangle((12.71-1.19,1.34-0.16), 1.14+1.19, 0.15+0.16, facecolor='mediumslateblue', fill=True, alpha=0.5)

LIGO_1 = Rectangle((10.8-1.7,1.36), 3.7, 1.62-1.36, facecolor='gold', fill=True, alpha=0.5)
LIGO_2 = Rectangle((10.7-1.5,1.15), 3.6, 1.36-1.15, facecolor='gold', fill=True, alpha=0.5)

#! m-r
plt.figure()
ax = plt.subplot(111)
ax.add_artist(NICER_Miller)
ax.add_artist(NICER_Riley)
ax.add_artist(LIGO_1)
ax.add_artist(LIGO_2)
for i in range(30):
    Tidal_file  = 'Tidal_out/SpecEOS/spec_'+str(i)+'_Tidal_data.txt'
    Tidal_data  = np.loadtxt(Tidal_file)
    Mass        = Tidal_data[:,2]
    Radius      = Tidal_data[:,3]
    index_max   = np.where(Mass==np.amax(Mass))[0][0]
    Mass        = Mass[:index_max+1]
    Radius      = Radius[:index_max+1]

    plt.plot(Radius, Mass, c=(0.3,0.5+0.01*i,1-0.01*i), linewidth=1)
# plt.plot(Radius_test, Mass_test, c='b', linewidth=1)
plt.text(9.4,1.3,'LIGO')
plt.text(8.5,1,'GW170817')
plt.text(13,1.4,'NICER')
plt.text(14,1.2,'PSR J0030+0451')
plt.xlim(8,20)
plt.ylim(0,3)
plt.xlabel('R[km]')
plt.ylabel(r'$M[M_\odot]$')
plt.grid(alpha=0.5)
plt.savefig('Tidal_out/SpecEOS/MR_Spec.jpg', dpi=400)

#! love-m
plt.figure()
for i in range(30):
    Tidal_file  = 'Tidal_out/SpecEOS/spec_'+str(i)+'_Tidal_data.txt'
    Tidal_data  = np.loadtxt(Tidal_file)
    Mass        = Tidal_data[:,2]
    Love        = Tidal_data[:,5]
    index_max   = np.where(Mass==np.amax(Mass))[0][0]
    Mass        = Mass[:index_max+1]
    Love        = Love[:index_max+1]
    
    plt.plot(Mass, Love, c=(0.3,0.5+0.01*i,1-0.01*i), linewidth=1)
# plt.plot(Mass_test, Love_test, c=(0.1,0.2,0.5), label='test', linewidth=1)
plt.yscale('log')
plt.xlim(0,2.2)
plt.ylim(1e1,1e9)
plt.xlabel(r'$M[M_\odot]$')
plt.ylabel(r'$\bar{\lambda}$')
plt.grid(alpha=0.5)
plt.savefig('Tidal_out/SpecEOS/LoveM_Spec.jpg', dpi=400)


#! love-c
plt.figure()
for i in range(30):
    Tidal_file  = 'Tidal_out/SpecEOS/spec_'+str(i)+'_Tidal_data.txt'
    Tidal_data  = np.loadtxt(Tidal_file)
    Mass        = Tidal_data[:,2]
    Compactness = Tidal_data[:,4]
    Love        = Tidal_data[:,5]
    index_max   = np.where(Mass==np.amax(Mass))[0][0]
    Compactness = Compactness[:index_max+1]
    Love        = Love[:index_max+1]

    plt.loglog(Compactness, Love, c=(0.3,0.5+0.01*i,1-0.01*i), linewidth=1)
# plt.loglog(Compactness_test, Love_test, c='black', label='test')
plt.xlim(4e-3,5e-1)
plt.ylim(1e-1,1e9)
plt.xlabel('C')
plt.ylabel(r'$\bar{\lambda}$')
plt.grid(alpha=0.5)
plt.savefig('Tidal_out/SpecEOS/LoveC_Spec.jpg', dpi=400)



#! dimlove-m
plt.figure()
for i in range(30):
    Tidal_file  = 'Tidal_out/SpecEOS/spec_'+str(i)+'_Tidal_data.txt'
    Tidal_data  = np.loadtxt(Tidal_file)
    Mass        = Tidal_data[:,2]
    Love        = Tidal_data[:,5]
    Love_d      = Love*((Mass*solar_mass)**5)
    Love_d      = Love_d*unitconv/1e36
    index_max   = np.where(Mass==np.amax(Mass))[0][0]
    Mass        = Mass[:index_max+1]
    Love_d      = Love_d[:index_max+1]

    plt.plot(Mass, Love_d, c=(0.3,0.5+0.01*i,1-0.01*i), linewidth=1)
# plt.plot(Mass_test, Love_test_d, c='black', label='test')

plt.xlim(0,3)
plt.ylim(0,10)
plt.xlabel(r'$M[M_\odot]$')
plt.ylabel(r'$\lambda[10^{36}g\cdot cm^2\cdot s^2]$')
plt.grid(alpha=0.5)
plt.savefig('Tidal_out/SpecEOS/dLoveM_Spec.jpg', dpi=400)