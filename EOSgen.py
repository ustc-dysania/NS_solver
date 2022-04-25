from math import gamma
import numpy as np
# from scipy.interpolate import interp1d
from scipy.integrate import quad
#g/cm^3
gravitational_constant = 6.6743e-11  # m^3 kg^-1 s^-2
speed_of_light = 299792458.0         # m/s


#? using sly4 as low density eos
# EOS_file = 'Spec_EOS/LALSimNeutronStarEOS_SLY4.dat'
# # unitconv = gravitational_constant*1e-3 / speed_of_light**2 * 1e-3 / (1e-5)**3 #convert from g/cm^3 to km^-2
# eos_data = np.loadtxt(EOS_file)
# pressure_list = eos_data[:,0] * 1e6          # convert to the unit of km^-2
# energy_density_list = eos_data[:,1] * 1e6    # convert to the unit of km^-2
# energydensity_of_pressure = interp1d(pressure_list, energy_density_list, fill_value='extrapolate')  # linear interpoltation by default 

#? using true eos as low density eos
EOS_file = 'EOS_Tables/sly.dat' 
unitconv = gravitational_constant*1e-3 / speed_of_light**2 * 1e-3 / (1e-5)**3 #convert from g/cm^3 to km^-2
eos_data = np.loadtxt(EOS_file)
pressure_list = eos_data[:,1] * unitconv          # convert to the unit of km^-2
energy_density_list = eos_data[:,2] * unitconv    # convert to the unit of km^-2
# energydensity_of_pressure = interp1d(pressure_list, energy_density_list, fill_value='extrapolate')  # linear interpoltation by default 



def integrand_M(x, gamma0, gamma1, gamma2, gamma3):
    Gamma = gamma0 + gamma1 * x + gamma2 * x**2 + gamma3 * x**3
    Gamma = np.exp(Gamma)
    integrand = 1. / Gamma

    return integrand

def mu_cal(x, gamma0, gamma1, gamma2, gamma3):
    mu = quad(integrand_M, a=0, b=x, args=(gamma0, gamma1, gamma2, gamma3))[0]
    mu = np.exp(-mu)
    return mu

def integrand_E(x, gamma0, gamma1, gamma2, gamma3):
    integrand = np.exp(x) * mu_cal(x, gamma0, gamma1, gamma2, gamma3) * integrand_M(x, gamma0, gamma1, gamma2, gamma3)
    return integrand



def energydensity_of_x(p0, e0, x, gamma0, gamma1, gamma2, gamma3):
    mu = mu_cal(x, gamma0, gamma1, gamma2, gamma3)
    T = quad(integrand_E, a=0, b=x, args=(gamma0, gamma1, gamma2, gamma3))[0]
    # e0 = energydensity_of_pressure(p0)
    energydensity = e0 / mu + p0 / mu * T

    return energydensity

def gamma0_dist(i):
    t = np.random.rand(i)
    gamma0 = t * (1.0215-0.8651)+0.8651         #! range of true eos passing through constraint
    # gamma0 = t * (1.8662-0.4132)+0.4132
    return gamma0

def gamma1_dist(i):
    t = np.random.rand(i)
    gamma1 = t * (0.2716-0.0656)+0.0656         #! range of true eos passing through constraint
    # gamma1 = t * (1.5237+1.4266)-1.4266
    return gamma1

def gamma2_dist(i):
    t = np.random.rand(i)
    gamma2 = t * (0.0765+0.0862)-0.0862         #! range of true eos passing through constraint
    # gamma2 = t * (0.4450+0.5817)-0.5817
    return gamma2

def gamma3_dist(i):
    t = np.random.rand(i)
    gamma3 = t * (0.0075+0.0177)-0.0177         #! range of true eos passing through constraint
    # gamma3 = t * (0.0571+0.0389)-0.0389
    return gamma3

def p0_dist(i):
    t = np.random.rand(i)
    p0 = t * (1.64-1.33)*1e33+1.33e33           #! range of true eos passing through constraint
    # p0 = t * (5.49-0.81)*1e33+0.81e33
    return p0

def e0_dist(i):
    t = np.random.rand(i)
    e0 = t * (2.05-2.02)*1e14+2.02e14           #! range of true eos passing through constraint
    # e0 = t * (2.06-2.02)*1e14+2.02e14
    return e0

if __name__ == "__main__":

    # p0 = 1.14e33 * 0.1 * gravitational_constant / speed_of_light ** 4 * 1e6
    # # p0 = 3.18e33                        
    # e0 = 2.04e14 * 1e4 * 0.1 * gravitational_constant / speed_of_light ** 2 * 1e6             #? convert cgs to km^-2
    # # e0 = 2.03e14                        
    # gamma0 = 0.6785
    # gamma1 = 0.2626
    # gamma2 = -0.0215
    # gamma3 = -0.0008

    # gamma0 = 1.2132
    # gamma1 = -0.0648
    # gamma2 = 0.0561
    # gamma3 = -0.0111

    
    # xmax = 7.48
    # xmax = 5.40


    xmax = 8.0


    npint = 1000



    i=21

    gamma0 = gamma0_dist(1)
    gamma1 = gamma1_dist(1)
    gamma2 = gamma2_dist(1)
    gamma3 = gamma3_dist(1)
    p0     = p0_dist(1)* 0.1 * gravitational_constant / speed_of_light ** 4 * 1e6
    e0     = e0_dist(1)* 1e4 * 0.1 * gravitational_constant / speed_of_light ** 2 * 1e6

    para = [gamma0, gamma1, gamma2, gamma3, p0, e0]
    np.savetxt('Spec_EOS/para_'+str(i)+'_.txt', para)


    eos_table = np.zeros((npint,2))                     #? first row is p second row is e

    x_range = np.linspace(0, xmax, npint)

    eos_table[:,0] = np.exp(x_range)*p0

    for j in range(0, len(x_range)):
        eos_table[j,1] = energydensity_of_x(p0, e0, x_range[j], gamma0, gamma1, gamma2, gamma3)


        # eos_table[:,1] = energydensity_of_x(p0, x_range, gamma0, gamma1, gamma2, gamma3)


    index = np.where((pressure_list < eos_table[0][0]) & (energy_density_list < eos_table[0][1]))

    # index = np.where(pressure_list < eos_table[0][0]) 
    eos_sly = np.stack((pressure_list[index], energy_density_list[index]),axis=-1)

    eos_table = np.vstack((eos_sly, eos_table))

    np.savetxt('Spec_EOS/spec_'+str(i)+'_.txt', eos_table)








    # para_table = np.zeros((30,6))
    # para_table[:,0] = gamma0_dist(30)
    # para_table[:,1] = gamma1_dist(30)
    # para_table[:,2] = gamma2_dist(30)
    # para_table[:,3] = gamma3_dist(30)
    # para_table[:,4] = p0_dist(30)
    # para_table[:,5] = e0_dist(30)

    # xmax = 8.0


    # npint = 1000


    # for i in range(30):
    #     gamma0 = para_table[i,0]
    #     gamma1 = para_table[i,1]
    #     gamma2 = para_table[i,2]
    #     gamma3 = para_table[i,3]
    #     p0     = para_table[i,4]* 0.1 * gravitational_constant / speed_of_light ** 4 * 1e6
    #     e0     = para_table[i,5]* 1e4 * 0.1 * gravitational_constant / speed_of_light ** 2 * 1e6


    #     eos_table = np.zeros((npint,2))                     #? first row is p second row is e

    #     x_range = np.linspace(0, xmax, npint)

    #     eos_table[:,0] = np.exp(x_range)*p0

    #     for j in range(0, len(x_range)):
    #         eos_table[j,1] = energydensity_of_x(p0, e0, x_range[j], gamma0, gamma1, gamma2, gamma3)


    #     # eos_table[:,1] = energydensity_of_x(p0, x_range, gamma0, gamma1, gamma2, gamma3)


    #     index = np.where((pressure_list < eos_table[0][0]) & (energy_density_list < eos_table[0][1]))

    #     # index = np.where(pressure_list < eos_table[0][0]) 
    #     eos_sly = np.stack((pressure_list[index], energy_density_list[index]),axis=-1)

    #     eos_table = np.vstack((eos_sly, eos_table))

    #     np.savetxt('Spec_EOS/spec_'+str(i)+'_.txt', eos_table)
