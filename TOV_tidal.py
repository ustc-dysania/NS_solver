import numpy as np
from numpy import pi, log
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

gravitational_constant = 6.6743e-11  # m^3 kg^-1 s^-2
speed_of_light = 299792458.0         # m/s
solar_mass = 1.9884099021470415e+30  # Kg
km2Msun = 1000*speed_of_light**2/gravitational_constant/solar_mass

#? read in the EoS data and interpolate
# eos = 'sly'
#eos = 'ap3'
#eos = 'eng'
#eos = 'alf2'
#eos = 'wff2'
#eos = 'ap4'
#eos = 'mpa1'
#eos = 'ms1'
# eos = 'ms1b'


# EOS_file = 'EOS_Tables/' + eos + '.dat' 
# unitconv = gravitational_constant*1e-3 / speed_of_light**2 * 1e-3 / (1e-5)**3 #convert from g/cm^3 to km^-2
# eos_data = np.loadtxt(EOS_file)
# pressure_list = eos_data[:,1] * unitconv          # convert to the unit of km^-2
# energy_density_list = eos_data[:,2] * unitconv    # convert to the unit of km^-2
# energydensity_of_pressure = interp1d(pressure_list, energy_density_list, fill_value='extrapolate')  # linear interpoltation by default 
# pressure_upperbound = max(pressure_list)
# pressure_lowerbound = min(pressure_list)
# print(pressure_lowerbound)
# print(pressure_upperbound)



def TOV_Tidal_eqns_DEF(r,y):
    m_of_r = y[0]   #! in unit of km^-2
    p_of_r = y[1]
    v_of_r = y[2]
    h_of_r = y[3]

    rho = energydensity_of_pressure(p_of_r)
    dnudr = (2*m_of_r+8*pi*r**3*p_of_r)/r/(r-2*m_of_r)

    try:
        dmdr = 4*pi*r**2*rho
        dpdr = -(rho+p_of_r)*(m_of_r+4*pi*r**3*p_of_r)/(r*(r-2*m_of_r))
        dvdr = -h_of_r*dnudr
        dhdr = (-dnudr+r/(r-2*m_of_r)/dnudr*(8*pi*(rho+p_of_r)-4*m_of_r/(r**3)))*h_of_r-4*v_of_r/r/(r-2*m_of_r)/dnudr
    except:
        raise Exception('fail to evaluate TOV_eqns at: r={}, m_of_r={},p_of_r={}'.format(r, m_of_r, p_of_r))
    return [dmdr, dpdr, dvdr, dhdr]

def reach_surface(r, y):
    return y[1]-1e-18
reach_surface.terminal = True
reach_surface.direction = 0

#! with M initial
def TOV_Tidal_eqns_sol(p_init, rho_init, h_init_const):
    if p_init>pressure_upperbound or p_init<pressure_lowerbound:
        print('warning'+'pressure initial value {} is outside of the EOS data range, '
                    'extrapolation will be used. If you see too much this warning, '
                    'you may need to adjust the prior'.format(p_init))
    r_init = 0.001
    m_init = 4/3*pi*r_init**3*rho_init
    h_init = h_init_const*r_init**2
    v_init = -2*pi*(p_init+rho_init/3)*h_init_const*r_init**4

    try:
        solution  = solve_ivp(fun=TOV_Tidal_eqns_DEF, t_span=(r_init, 1000.0), y0=[m_init, p_init, v_init, h_init], 
                              events=reach_surface, rtol=1e-10, atol=1e-18)
    except:
        print('error '+'fail to call solve_ivp() at: ')
        raise Exception('p_init={}'.format(p_init))
    if solution.success == False:
        print('error '+'Failing to solve TOV equations at: ')
        raise Exception('p_init={}'.format(p_init))
    if solution.status == 0:
        print('error '+'the solution didn\'t reach the surface')
        raise Exception('p_init={}'.format(p_init))

    radius = solution.t_events[0][0]
    m_surface = solution.y_events[0][0][0]
    #p_surface = solution.y_events[0][0][1]
    h_surface = solution.y_events[0][0][3]

    dydr_events = TOV_Tidal_eqns_DEF(radius, solution.y_events[0][0])
    #dmdr_surface = dydr_events[0]
    #dpdr_surface = dydr_events[1]
    dhdr_surface = dydr_events[3]

    #? mass in km
    mass = m_surface
    compactness = mass/radius
    y = radius*dhdr_surface/h_surface

    Love = 8/15*2*(1-2*compactness)**2*(2+2*compactness*(y-1)-y)/\
           (2*compactness*(6-3*y+3*compactness*(5*y-8))
           +4*compactness**3*(13-11*y+compactness*(3*y-2)+2*compactness**2*(1+y))
           +3*(1-2*compactness)**2*(2-y+2*compactness*(y-1))*log(1-2*compactness))

    mass = mass*km2Msun

    solution_dict = dict(mass=mass, radius=radius, compactness=compactness, Love=Love)

    return solution_dict

if __name__ == "__main__":

    #? pay attention to the range of p_c, the pressure is largest at the centor in a star, so the minimum value of the range
    #? should not be the minimum allowed value of the EOS, but the largest value allowed could be chosen as upper limit
    #p_init_ls = np.linspace(1e-6, pressure_upperbound, num=2000)


    # log_p_u = np.log10(pressure_upperbound)

    # p_init_ls = np.logspace(-6, log_p_u, num=5000)

    # sol_array = []

    h_init_const = 1e4

    # for p_init in p_init_ls:
    #     rho_init = energydensity_of_pressure(p_init)
    #     sol = TOV_Tidal_eqns_sol(p_init=p_init, rho_init=rho_init, h_init_const=h_init_const)  #! with M initial
    #     sol_item = [p_init, rho_init, sol['mass'], sol['radius'], sol['compactness'], sol['Love']]

    #     sol_array.append(sol_item)

    # sol_array = np.array(sol_array)
    # np.savetxt('Tidal_out/'+eos+'_Tidal_data.txt',sol_array)

    
    for i in range(30):
        EOS_table = np.loadtxt('Spec_EOS/spec_'+str(i)+'_.txt')
        pressure_list = EOS_table[:,0]
        energy_density_list = EOS_table[:,1]
        energydensity_of_pressure = interp1d(pressure_list, energy_density_list, fill_value='extrapolate')  # linear interpoltation by default 
        pressure_upperbound = max(pressure_list)
        pressure_lowerbound = min(pressure_list)


        log_p_u = np.log10(pressure_upperbound)
        p_init_ls = np.logspace(-6, log_p_u, num=5000)
    
        sol_array = []
        for p_init in p_init_ls:
            rho_init = energydensity_of_pressure(p_init)
            sol = TOV_Tidal_eqns_sol(p_init=p_init, rho_init=rho_init, h_init_const=h_init_const)  #! with M initial
            sol_item = [p_init, rho_init, sol['mass'], sol['radius'], sol['compactness'], sol['Love']]

            sol_array.append(sol_item)

        sol_array = np.array(sol_array)

        np.savetxt('Tidal_out/SpecEOS/spec_'+str(i)+'_Tidal_data.txt',sol_array)
    
    
    
    
    # for eos in ['bsk20','bsk21']:
    #     EOS_file = 'EOS_Tables/' + eos + '.dat' 
    #     unitconv = gravitational_constant*1e-3 / speed_of_light**2 * 1e-3 / (1e-5)**3 #convert from g/cm^3 to km^-2
    #     eos_data = np.loadtxt(EOS_file)
    #     # pressure_list = eos_data[:,1] * unitconv          # convert to the unit of km^-2
    #     # energy_density_list = eos_data[:,2] * unitconv    # convert to the unit of km^-2
    #     pressure_list = eos_data[:,2] * unitconv          # convert to the unit of km^-2    #! for bsk
    #     energy_density_list = eos_data[:,1] * unitconv    # convert to the unit of km^-2
    #     energydensity_of_pressure = interp1d(pressure_list, energy_density_list, fill_value='extrapolate')  # linear interpoltation by default 
    #     pressure_upperbound = max(pressure_list)
    #     pressure_lowerbound = min(pressure_list)


    #     #p_init_ls = np.linspace(1e-6, pressure_upperbound, num=20000)
    #     log_p_u = np.log10(pressure_upperbound)
    #     # if pressure_lowerbound < 1e-6:
    #     #     p_init_ls = np.logspace(-6, log_p_u, num=5000)
    #     # else:
    #     #     p_init_ls = np.logspace(-5.5, log_p_u, num=5000)

    #     p_init_ls = np.logspace(-6, log_p_u, num=5000)
        
    #     sol_array = []

    #     for p_init in p_init_ls:
    #         rho_init = energydensity_of_pressure(p_init)
    #         sol = TOV_Tidal_eqns_sol(p_init=p_init, rho_init=rho_init, h_init_const=h_init_const)  #! with M initial
    #         sol_item = [p_init, rho_init, sol['mass'], sol['radius'], sol['compactness'], sol['Love']]

    #         sol_array.append(sol_item)

    #     sol_array = np.array(sol_array)
    #     np.savetxt('Tidal_out/'+eos+'_Tidal_data.txt',sol_array)





#! dasd
#? dsad
#//  dsadd
#todo dsadasdsd
#* dsadasda