import xtrack as xt

import lumi

collider = xt.Multiline.from_json('../hllhc15_collision/collider_02_tuned_bb_off.json')
collider.build_trackers()

n_colliding_bunches = 2808
num_particles_per_bunch = 1.15e11
ip_name = 'ip8'
nemitt_x = 3.75e-6
nemitt_y = 3.75e-6
sigma_z = 0.0755

twb1 = collider.lhcb1.twiss()
twb2 = collider.lhcb2.twiss().reverse()

lumi = lumi.luminosity(
    f=1/twb1.T_rev0,
    rest_mass_b1=twb1.particle_on_co.mass0 * 1e-9, # GeV
    rest_mass_b2=twb2.particle_on_co.mass0 * 1e-9, # GeV
    nb=n_colliding_bunches,
    N1=num_particles_per_bunch,
    N2=num_particles_per_bunch,
    x_1=twb1['x', ip_name],
    x_2=twb2['x', ip_name],
    y_1=twb1['y', ip_name],
    y_2=twb2['y', ip_name],
    px_1=twb1['px', ip_name],
    px_2=twb2['px', ip_name],
    py_1=twb1['py', ip_name],
    py_2=twb2['py', ip_name],
    energy_tot1=twb1.particle_on_co.energy0[0]*1e-9, # GeV
    energy_tot2=twb2.particle_on_co.energy0[0]*1e-9, # GeV
    deltap_p0_1=0, # energy spread (for now we neglect effect of dispersion)
    deltap_p0_2=0, # energy spread (for now we neglect effect of dispersion)
    epsilon_x1=nemitt_x,
    epsilon_x2=nemitt_x,
    epsilon_y1=nemitt_y,
    epsilon_y2=nemitt_y,
    sigma_z1=sigma_z,
    sigma_z2=sigma_z,
    beta_x1=twb1['betx', ip_name],
    beta_x2=twb2['betx', ip_name],
    beta_y1=twb1['bety', ip_name],
    beta_y2=twb2['bety', ip_name],
    alpha_x1=twb1['alfx', ip_name],
    alpha_x2=twb2['alfx', ip_name],
    alpha_y1=twb1['alfy', ip_name],
    alpha_y2=twb2['alfy', ip_name],
    dx_1=twb1['dx', ip_name],
    dx_2=twb2['dx', ip_name],
    dy_1=twb1['dy', ip_name],
    dy_2=twb2['dy', ip_name],
    dpx_1=twb1['dpx', ip_name],
    dpx_2=twb2['dpx', ip_name],
    dpy_1=twb1['dpy', ip_name],
    dpy_2=twb2['dpy', ip_name],
    CC_V_x_1=0, CC_f_x_1=0, CC_phase_x_1=0,
    CC_V_x_2=0, CC_f_x_2=0, CC_phase_x_2=0,
    CC_V_y_1=0, CC_f_y_1=0, CC_phase_y_1=0,
    CC_V_y_2=0, CC_f_y_2=0, CC_phase_y_2=0,
    R12_1=0, R22_1=0, R34_1=0, R44_1=0,
    R12_2=0, R22_2=0, R34_2=0, R44_2=0,
    verbose=False, sigma_integration=3)
