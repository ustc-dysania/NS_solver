from matplotlib import pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

gravitational_constant = 6.6743e-11  # m^3 kg^-1 s^-2
speed_of_light = 299792458.0         # m/s
solar_mass = 1.9884099021470415e+30  # Kg


# for eos in ['sly','ap3','eng','alf2','bsk21','H4','mpa1']:

#? Read in the Tidal data 
eos = 'sly'
Tidal_file = 'Tidal_out/' + eos + '_Tidal_data.txt'

Tidal_data = np.loadtxt(Tidal_file)

p_c_sly         = Tidal_data[:,0]
rho_c_sly       = Tidal_data[:,1]
Mass_sly        = Tidal_data[:,2]
Radius_sly      = Tidal_data[:,3]
Compactness_sly = Tidal_data[:,4]
Love_sly        = Tidal_data[:,5]
Love_sly_d      = Love_sly*((Mass_sly*solar_mass)**5)



eos = 'ap3'
Tidal_file = 'Tidal_out/' + eos + '_Tidal_data.txt'

Tidal_data = np.loadtxt(Tidal_file)

p_c_ap3         = Tidal_data[:,0]
rho_c_ap3       = Tidal_data[:,1]
Mass_ap3        = Tidal_data[:,2]
Radius_ap3      = Tidal_data[:,3]
Compactness_ap3 = Tidal_data[:,4]
Love_ap3        = Tidal_data[:,5]
Love_ap3_d      = Love_ap3*((Mass_ap3*solar_mass)**5)



eos = 'eng'
Tidal_file = 'Tidal_out/' + eos + '_Tidal_data.txt'

Tidal_data = np.loadtxt(Tidal_file)

p_c_eng         = Tidal_data[:,0]
rho_c_eng       = Tidal_data[:,1]
Mass_eng        = Tidal_data[:,2]
Radius_eng      = Tidal_data[:,3]
Compactness_eng = Tidal_data[:,4]
Love_eng        = Tidal_data[:,5]
Love_eng_d      = Love_eng*((Mass_eng*solar_mass)**5)



eos = 'alf2'
Tidal_file = 'Tidal_out/' + eos + '_Tidal_data.txt'

Tidal_data = np.loadtxt(Tidal_file)

p_c_alf2         = Tidal_data[:,0]
rho_c_alf2       = Tidal_data[:,1]
Mass_alf2        = Tidal_data[:,2]
Radius_alf2      = Tidal_data[:,3]
Compactness_alf2 = Tidal_data[:,4]
Love_alf2        = Tidal_data[:,5]
Love_alf2_d      = Love_alf2*((Mass_alf2*solar_mass)**5)



eos = 'alf4'
Tidal_file = 'Tidal_out/' + eos + '_Tidal_data.txt'

Tidal_data = np.loadtxt(Tidal_file)

p_c_alf4         = Tidal_data[:,0]
rho_c_alf4       = Tidal_data[:,1]
Mass_alf4        = Tidal_data[:,2]
Radius_alf4      = Tidal_data[:,3]
Compactness_alf4 = Tidal_data[:,4]
Love_alf4        = Tidal_data[:,5]
Love_alf4_d      = Love_alf4*((Mass_alf4*solar_mass)**5)



eos = 'wff1'
Tidal_file = 'Tidal_out/' + eos + '_Tidal_data.txt'

Tidal_data = np.loadtxt(Tidal_file)

p_c_wff1         = Tidal_data[:,0]
rho_c_wff1       = Tidal_data[:,1]
Mass_wff1        = Tidal_data[:,2]
Radius_wff1      = Tidal_data[:,3]
Compactness_wff1 = Tidal_data[:,4]
Love_wff1        = Tidal_data[:,5]
Love_wff1_d      = Love_wff1*((Mass_wff1*solar_mass)**5)



eos = 'wff2'
Tidal_file = 'Tidal_out/' + eos + '_Tidal_data.txt'

Tidal_data = np.loadtxt(Tidal_file)

p_c_wff2         = Tidal_data[:,0]
rho_c_wff2       = Tidal_data[:,1]
Mass_wff2        = Tidal_data[:,2]
Radius_wff2      = Tidal_data[:,3]
Compactness_wff2 = Tidal_data[:,4]
Love_wff2        = Tidal_data[:,5]
Love_wff2_d      = Love_wff2*((Mass_wff2*solar_mass)**5)


eos = 'ap4'
Tidal_file = 'Tidal_out/' + eos + '_Tidal_data.txt'

Tidal_data = np.loadtxt(Tidal_file)

p_c_ap4         = Tidal_data[:,0]
rho_c_ap4       = Tidal_data[:,1]
Mass_ap4        = Tidal_data[:,2]
Radius_ap4      = Tidal_data[:,3]
Compactness_ap4 = Tidal_data[:,4]
Love_ap4        = Tidal_data[:,5]
Love_ap4_d      = Love_ap4*((Mass_ap4*solar_mass)**5)



eos = 'mpa1'
Tidal_file = 'Tidal_out/' + eos + '_Tidal_data.txt'

Tidal_data = np.loadtxt(Tidal_file)

p_c_mpa1         = Tidal_data[:,0]
rho_c_mpa1       = Tidal_data[:,1]
Mass_mpa1        = Tidal_data[:,2]
Radius_mpa1      = Tidal_data[:,3]
Compactness_mpa1 = Tidal_data[:,4]
Love_mpa1        = Tidal_data[:,5]
Love_mpa1_d      = Love_mpa1*((Mass_mpa1*solar_mass)**5)



eos = 'gnh3'
Tidal_file = 'Tidal_out/' + eos + '_Tidal_data.txt'

Tidal_data = np.loadtxt(Tidal_file)

p_c_gnh3         = Tidal_data[:,0]
rho_c_gnh3       = Tidal_data[:,1]
Mass_gnh3        = Tidal_data[:,2]
Radius_gnh3      = Tidal_data[:,3]
Compactness_gnh3 = Tidal_data[:,4]
Love_gnh3        = Tidal_data[:,5]
Love_gnh3_d      = Love_gnh3*((Mass_gnh3*solar_mass)**5)

eos = 'ms1'
Tidal_file = 'Tidal_out/' + eos + '_Tidal_data.txt'

Tidal_data = np.loadtxt(Tidal_file)

p_c_ms1         = Tidal_data[:,0]
rho_c_ms1       = Tidal_data[:,1]
Mass_ms1        = Tidal_data[:,2]
Radius_ms1      = Tidal_data[:,3]
Compactness_ms1 = Tidal_data[:,4]
Love_ms1        = Tidal_data[:,5]
Love_ms1_d      = Love_ms1*((Mass_ms1*solar_mass)**5)



eos = 'ms1b'
Tidal_file = 'Tidal_out/' + eos + '_Tidal_data.txt'

Tidal_data = np.loadtxt(Tidal_file)

p_c_ms1b         = Tidal_data[:,0]
rho_c_ms1b       = Tidal_data[:,1]
Mass_ms1b        = Tidal_data[:,2]
Radius_ms1b      = Tidal_data[:,3]
Compactness_ms1b = Tidal_data[:,4]
Love_ms1b        = Tidal_data[:,5]
Love_ms1b_d      = Love_ms1b*((Mass_ms1b*solar_mass)**5)

eos = 'bsk20'
Tidal_file = 'Tidal_out/' + eos + '_Tidal_data.txt'

Tidal_data = np.loadtxt(Tidal_file)

p_c_bsk20         = Tidal_data[:,0]
rho_c_bsk20       = Tidal_data[:,1]
Mass_bsk20        = Tidal_data[:,2]
Radius_bsk20      = Tidal_data[:,3]
Compactness_bsk20 = Tidal_data[:,4]
Love_bsk20        = Tidal_data[:,5]
Love_bsk20_d      = Love_bsk20*((Mass_bsk20*solar_mass)**5)


eos = 'bsk21'
Tidal_file = 'Tidal_out/' + eos + '_Tidal_data.txt'

Tidal_data = np.loadtxt(Tidal_file)

p_c_bsk21         = Tidal_data[:,0]
rho_c_bsk21       = Tidal_data[:,1]
Mass_bsk21        = Tidal_data[:,2]
Radius_bsk21      = Tidal_data[:,3]
Compactness_bsk21 = Tidal_data[:,4]
Love_bsk21        = Tidal_data[:,5]
Love_bsk21_d      = Love_bsk21*((Mass_bsk21*solar_mass)**5)




#? get love number to 10^36 g cm^2 s^2
unitconv      = gravitational_constant**4/speed_of_light**10*1e3*1e4         #? convert kg^5 to g cm^2 s^2
Love_sly_d    = Love_sly_d*unitconv/1e36
Love_ap4_d    = Love_ap4_d*unitconv/1e36
Love_ap3_d    = Love_ap3_d*unitconv/1e36
Love_eng_d    = Love_eng_d*unitconv/1e36
Love_alf2_d   = Love_alf2_d*unitconv/1e36
Love_alf4_d   = Love_alf4_d*unitconv/1e36
Love_gnh3_d   = Love_gnh3_d*unitconv/1e36
Love_wff1_d   = Love_wff1_d*unitconv/1e36
Love_wff2_d   = Love_wff2_d*unitconv/1e36
Love_ap4_d    = Love_ap4_d*unitconv/1e36
Love_mpa1_d   = Love_mpa1_d*unitconv/1e36
Love_ms1_d    = Love_ms1_d*unitconv/1e36
Love_ms1b_d   = Love_ms1b_d*unitconv/1e36
Love_bsk20_d  = Love_bsk20_d*unitconv/1e36
Love_bsk21_d  = Love_bsk21_d*unitconv/1e36




width_NICER = 14.255-11.959
height_NICER = 1.594-1.299
NICER_Miller = Rectangle((11.959,1.299), width_NICER, height_NICER, facecolor='violet', fill=True, alpha=0.5)

NICER_Riley = Rectangle((12.71-1.19,1.34-0.16), 1.14+1.19, 0.15+0.16, facecolor='mediumslateblue', fill=True, alpha=0.5)

LIGO_1 = Rectangle((10.8-1.7,1.36), 3.7, 1.62-1.36, facecolor='gold', fill=True, alpha=0.5)
LIGO_2 = Rectangle((10.7-1.5,1.15), 3.6, 1.36-1.15, facecolor='gold', fill=True, alpha=0.5)


#* remove the part where the mass begins to drop after reaching maximum mass
index_max         = np.where(Mass_alf2==np.amax(Mass_alf2))[0][0]
Mass_alf2         = Mass_alf2[:index_max+1]
Radius_alf2       = Radius_alf2[:index_max+1]
Compactness_alf2  = Compactness_alf2[:index_max+1]
Love_alf2         = Love_alf2[:index_max+1]
Love_alf2_d       = Love_alf2_d[:index_max+1]

index_max         = np.where(Mass_alf4==np.amax(Mass_alf4))[0][0]
Mass_alf4         = Mass_alf4[:index_max+1]
Radius_alf4       = Radius_alf4[:index_max+1]
Compactness_alf4  = Compactness_alf4[:index_max+1]
Love_alf4         = Love_alf4[:index_max+1]
Love_alf4_d       = Love_alf4_d[:index_max+1]

index_max         = np.where(Mass_ap3==np.amax(Mass_ap3))[0][0]
Mass_ap3          = Mass_ap3[:index_max+1]
Radius_ap3        = Radius_ap3[:index_max+1]
Compactness_ap3   = Compactness_ap3[:index_max+1]
Love_ap3          = Love_ap3[:index_max+1]
Love_ap3_d        = Love_ap3_d[:index_max+1]

index_max         = np.where(Mass_ap4==np.amax(Mass_ap4))[0][0]
Mass_ap4          = Mass_ap4[:index_max+1]
Radius_ap4        = Radius_ap4[:index_max+1]
Compactness_ap4   = Compactness_ap4[:index_max+1]
Love_ap4          = Love_ap4[:index_max+1]
Love_ap4_d        = Love_ap4_d[:index_max+1]

index_max         = np.where(Mass_bsk20==np.amax(Mass_bsk20))[0][0]
Mass_bsk20        = Mass_bsk20[:index_max+1]
Radius_bsk20      = Radius_bsk20[:index_max+1]
Compactness_bsk20 = Compactness_bsk20[:index_max+1]
Love_bsk20        = Love_bsk20[:index_max+1]
Love_bsk20_d      = Love_bsk20_d[:index_max+1]

index_max         = np.where(Mass_bsk21==np.amax(Mass_bsk21))[0][0]
Mass_bsk21        = Mass_bsk21[:index_max+1]
Radius_bsk21      = Radius_bsk21[:index_max+1]
Compactness_bsk21 = Compactness_bsk21[:index_max+1]
Love_bsk21        = Love_bsk21[:index_max+1]
Love_bsk21_d      = Love_bsk21_d[:index_max+1]

index_max        = np.where(Mass_eng==np.amax(Mass_eng))[0][0]
Mass_eng         = Mass_eng[:index_max+1]
Radius_eng       = Radius_eng[:index_max+1]
Compactness_eng  = Compactness_eng[:index_max+1]
Love_eng         = Love_eng[:index_max+1]
Love_eng_d       = Love_eng_d[:index_max+1]

index_max        = np.where(Mass_gnh3==np.amax(Mass_gnh3))[0][0]
Mass_gnh3        = Mass_gnh3[:index_max+1]
Radius_gnh3      = Radius_gnh3[:index_max+1]
Compactness_gnh3 = Compactness_gnh3[:index_max+1]
Love_gnh3        = Love_gnh3[:index_max+1]
Love_gnh3_d      = Love_gnh3_d[:index_max+1]

index_max        = np.where(Mass_mpa1==np.amax(Mass_mpa1))[0][0]
Mass_mpa1        = Mass_mpa1[:index_max+1]
Radius_mpa1      = Radius_mpa1[:index_max+1]
Compactness_mpa1 = Compactness_mpa1[:index_max+1]
Love_mpa1        = Love_mpa1[:index_max+1]
Love_mpa1_d      = Love_mpa1_d[:index_max+1]

index_max        = np.where(Mass_ms1==np.amax(Mass_ms1))[0][0]
Mass_ms1         = Mass_ms1[:index_max+1]
Radius_ms1       = Radius_ms1[:index_max+1]
Compactness_ms1  = Compactness_ms1[:index_max+1]
Love_ms1         = Love_ms1[:index_max+1]
Love_ms1_d       = Love_ms1_d[:index_max+1]

index_max        = np.where(Mass_ms1b==np.amax(Mass_ms1b))[0][0]
Mass_ms1b        = Mass_ms1b[:index_max+1]
Radius_ms1b      = Radius_ms1b[:index_max+1]
Compactness_ms1b = Compactness_ms1b[:index_max+1]
Love_ms1b        = Love_ms1b[:index_max+1]
Love_ms1b_d      = Love_ms1b_d[:index_max+1]

index_max        = np.where(Mass_sly==np.amax(Mass_sly))[0][0]
Mass_sly         = Mass_sly[:index_max+1]
Radius_sly       = Radius_sly[:index_max+1]
Compactness_sly  = Compactness_sly[:index_max+1]
Love_sly         = Love_sly[:index_max+1]
Love_sly_d       = Love_sly_d[:index_max+1]

index_max        = np.where(Mass_wff1==np.amax(Mass_wff1))[0][0]
Mass_wff1        = Mass_wff1[:index_max+1]
Radius_wff1      = Radius_wff1[:index_max+1]
Compactness_wff1 = Compactness_wff1[:index_max+1]
Love_wff1        = Love_wff1[:index_max+1]
Love_wff1_d      = Love_wff1_d[:index_max+1]

index_max        = np.where(Mass_wff2==np.amax(Mass_wff2))[0][0]
Mass_wff2        = Mass_wff2[:index_max+1]
Radius_wff2      = Radius_wff2[:index_max+1]
Compactness_wff2 = Compactness_wff2[:index_max+1]
Love_wff2        = Love_wff2[:index_max+1]
Love_wff2_d      = Love_wff2_d[:index_max+1]


#! special operation on bsk21 for the upgoing tail
index_bsk21 = np.where(Mass_bsk21==Mass_bsk21.min())[0][0]

Mass_bsk21 = Mass_bsk21[index_bsk21:]
Radius_bsk21 = Radius_bsk21[index_bsk21:]
Compactness_bsk21 = Compactness_bsk21[index_bsk21:]
Love_bsk21 = Love_bsk21[index_bsk21:]
Love_bsk21_d = Love_bsk21_d[index_bsk21:]


plt.figure()

ax = plt.subplot(111)

ax.add_artist(NICER_Miller)
ax.add_artist(NICER_Riley)
ax.add_artist(LIGO_1)
ax.add_artist(LIGO_2)
    

plt.plot(Radius_alf2, Mass_alf2, c='gold', label='alf2', linewidth=1)
plt.plot(Radius_alf4, Mass_alf4, c='tomato', label='alf4', linewidth=1)
plt.plot(Radius_ap3, Mass_ap3, c='red', label='ap3', linewidth=1)
plt.plot(Radius_ap4, Mass_ap4, c='turquoise',label='ap4', linewidth=1)
plt.plot(Radius_bsk20, Mass_bsk20, c='cadetblue',label='bsk20', linewidth=1)
plt.plot(Radius_bsk21, Mass_bsk21, c='lightpink',label='bsk21', linewidth=1)
plt.plot(Radius_eng, Mass_eng, c='deepskyblue', label='eng', linewidth=1)
plt.plot(Radius_gnh3, Mass_gnh3, c='cornflowerblue', label='gnh3', linewidth=1)
plt.plot(Radius_mpa1, Mass_mpa1, c='deeppink', label='mpa1', linewidth=1)
plt.plot(Radius_ms1, Mass_ms1, c='indigo', label='ms1', linewidth=1)
plt.plot(Radius_ms1b, Mass_ms1b, c='slateblue', label='ms1b', linewidth=1)
plt.plot(Radius_sly, Mass_sly, c='limegreen', label='sly', linewidth=1)
plt.plot(Radius_wff1, Mass_wff1, c='violet', label='wff1', linewidth=1)
plt.plot(Radius_wff2, Mass_wff2, c='chocolate', label='wff2', linewidth=1)



plt.text(9.4,1.3,'LIGO')
plt.text(8.5,1,'GW170817')
plt.text(13,1.4,'NICER')
plt.text(14,1.2,'PSR J0030+0451')





plt.xlim(8,20)
plt.ylim(0,3)
plt.xlabel('R[km]')
plt.ylabel(r'$M[M_\odot]$')
plt.grid(alpha=0.5)
plt.legend()
plt.savefig('Tidal_out/MR.jpg', dpi=400)


plt.figure()

plt.plot(Mass_alf2, Love_alf2, c='gold', label='alf2', linewidth=1)
plt.plot(Mass_alf4, Love_alf4, c='tomato', label='alf4', linewidth=1)
plt.plot(Mass_ap3, Love_ap3, c='red', label='ap3', linewidth=1)
plt.plot(Mass_ap4, Love_ap4, c='turquoise', label='ap4', linewidth=1)
plt.plot(Mass_bsk20, Love_bsk20, c='cadetblue', label='bsk20', linewidth=1)
plt.plot(Mass_bsk21, Love_bsk21, c='lightpink', label='bsk21', linewidth=1)
plt.plot(Mass_eng, Love_eng, c='deepskyblue', label='eng', linewidth=1)
plt.plot(Mass_gnh3, Love_gnh3, c='cornflowerblue', label='gnh3', linewidth=1)
plt.plot(Mass_mpa1, Love_mpa1, c='deeppink', label='mpa1', linewidth=1)
plt.plot(Mass_ms1, Love_ms1, c='indigo', label='ms1', linewidth=1)
plt.plot(Mass_ms1b, Love_ms1b, c='slateblue', label='ms1b', linewidth=1)
plt.plot(Mass_sly, Love_sly, c='limegreen', label='sly', linewidth=1)
plt.plot(Mass_wff1, Love_wff1, c='violet', label='wff1', linewidth=1)
plt.plot(Mass_wff2, Love_wff2, c='chocolate', label='wff2', linewidth=1)


plt.yscale('log')
plt.xlim(0,2.2)
plt.ylim(1e1,1e9)
plt.xlabel(r'$M[M_\odot]$')
plt.ylabel(r'$\bar{\lambda}$')
plt.grid(alpha=0.5)
plt.legend()
plt.savefig('Tidal_out/LoveM.jpg', dpi=400)

plt.figure()

plt.loglog(Compactness_alf2, Love_alf2, c='gold', label='alf2')
plt.loglog(Compactness_alf4, Love_alf4, c='tomato', label='alf4')
plt.loglog(Compactness_ap3, Love_ap3, c='red', label='ap3')
plt.loglog(Compactness_ap4, Love_ap4, c='turquoise', label='ap4')
plt.loglog(Compactness_bsk20, Love_bsk20, c='cadetblue', label='bsk20')
plt.loglog(Compactness_bsk21, Love_bsk21, c='lightpink', label='bsk21')
plt.loglog(Compactness_eng, Love_eng, c='deepskyblue', label='eng')
plt.loglog(Compactness_gnh3, Love_gnh3, c='cornflowerblue', label='gnh3')
plt.loglog(Compactness_mpa1, Love_mpa1, c='deeppink', label='mpa1')
plt.loglog(Compactness_ms1, Love_ms1, c='indigo', label='ms1')
plt.loglog(Compactness_ms1b, Love_ms1b, c='slateblue', label='ms1b')
plt.loglog(Compactness_sly, Love_sly, c='limegreen',label='sly')
plt.loglog(Compactness_wff1, Love_wff1, c='violet', label='wff1')
plt.loglog(Compactness_wff2, Love_wff2, c='chocolate', label='wff2')



plt.xlim(4e-3,5e-1)
plt.ylim(1e-1,1e9)
plt.xlabel('C')
plt.ylabel(r'$\bar{\lambda}$')
plt.grid(alpha=0.5)
plt.legend()
plt.savefig('Tidal_out/LoveC.jpg', dpi=400)




plt.figure()

plt.plot(Mass_alf2, Love_alf2_d, c='gold', label='alf2')
plt.plot(Mass_alf4, Love_alf4_d, c='tomato', label='alf4')
plt.plot(Mass_ap3, Love_ap3_d, c='red', label='ap3')
plt.plot(Mass_ap4, Love_ap4_d, c='turquoise', label='ap4')
plt.plot(Mass_bsk20, Love_bsk20_d, c='cadetblue', label='bsk20')
plt.plot(Mass_bsk21, Love_bsk21_d, c='lightpink', label='bsk21')
plt.plot(Mass_eng, Love_eng_d, c='deepskyblue', label='eng')
plt.plot(Mass_gnh3, Love_gnh3_d, c='cornflowerblue', label='gnh3')
plt.plot(Mass_mpa1, Love_mpa1_d, c='deeppink', label='mpa1')
plt.plot(Mass_ms1, Love_ms1_d, c='indigo', label='ms1')
plt.plot(Mass_ms1b, Love_ms1b_d, c='slateblue', label='ms1b')
plt.plot(Mass_sly, Love_sly_d, c='limegreen', label='sly')
plt.plot(Mass_wff1, Love_wff1_d, c='violet', label='wff1')
plt.plot(Mass_wff2, Love_wff2_d, c='chocolate', label='wff2')



plt.xlim(0,3)
plt.ylim(0,10)
plt.xlabel(r'$M[M_\odot]$')
plt.ylabel(r'$\lambda[10^{36}g\cdot cm^2\cdot s^2]$')
plt.grid(alpha=0.5)
plt.legend()
plt.savefig('Tidal_out/dLoveM.jpg', dpi=400)