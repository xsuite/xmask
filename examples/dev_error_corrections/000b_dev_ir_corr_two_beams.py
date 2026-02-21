import xtrack as xt
from integral_correction import IntegralCorrection

env = xt.load('collider_00_from_mad_with_errors.json')

# Reference twiss
env_no_err = xt.load('collider_00_from_mad_no_errors.json')
tw_b1 = env_no_err.lhcb1.twiss4d(reverse=False) # Reference twiss
tw_b2 = env_no_err.lhcb2.twiss4d(reverse=False) # Reference twiss


# Normal sextupole correction
# correction_knobs=['kcsx3.l5', 'kcsx3.r5']
# multipole='k2l'
# target_quantities={'c12': (1, 2, 'diff'), 'c21': (2, 1, 'diff')}

# Normal octupole correction ip5
generated_knob_name='on_corr_k3_ip5'
correction_knobs=['kcox3.l5', 'kcox3.r5']
multipole='k3l'
target_quantities_b1={'f4000_b1': 'f4000'}
target_quantities_b2={'f4000_b2': 'f4000'}
range_b1 = 'dfxj.4l5', 'dfxj.4r5'
range_b2 = 'dfxj.4r5', 'dfxj.4l5'
feed_down = False # to have same result as legacy

# Usage:
rdt_contrib_b1 = IntegralCorrection(
                         line=env['lhcb1'],
                         tw=tw_b1,
                         start=range_b1[0],
                         end=range_b1[1],
                         correction_knobs=correction_knobs,
                         multipole=multipole,
                         ip=None, # not needed when RDT are used
                         feed_down=feed_down,
                         target_quantities=target_quantities_b1,
                         generated_knob_name=generated_knob_name)

rdt_contrib_b2 = IntegralCorrection(
                         line=env['lhcb2'],
                         tw=tw_b2,
                         start=range_b2[0],
                         end=range_b2[1],
                         correction_knobs=[], # only targets here
                         multipole=multipole,
                         ip=None, # not needed when RDT are used
                         feed_down=feed_down,
                         target_quantities=target_quantities_b2,
                         generated_knob_name=generated_knob_name)

# knob_opt_b1 = rdt_contrib_b1.get_optimizer()
# knob_opt_b2 = rdt_contrib_b2.get_optimizer()

# combined_opt = knob_opt_b1.opt.clone(add_targets=knob_opt_b2.opt.targets)
# combined_opt.step()

# knob_opt_b1.generate_knob()

print("Original correction:")
rdt_contrib_b1.print_corrections()

rdt_contrib_b1.clear_corrections()

knob_opt_b1 = rdt_contrib_b1.get_optimizer()
knob_opt_b2 = rdt_contrib_b2.get_optimizer()

# opt = rdt_contrib_b1.correct() # correct only b1
combined_opt = knob_opt_b1.opt.clone(add_targets=knob_opt_b2.opt.targets)
combined_opt.step()
knob_opt_b1.generate_knob()



print("Before setting the knob:")
rdt_contrib_b1.print_corrections()

env[knob_opt_b1.knob_name] = 1.0
print("After setting the knob:")
rdt_contrib_b1.print_corrections()

# oo = rdt_contrib.run()
# integrand_rdt = oo['(0, 4)_integrand_rdt']
# integrand_loc = oo['(0, 4)_integrand_loc']
# s = oo['s']
# integrand_rdt *= np.exp(-1j * np.angle(integrand_rdt[0]))

# import matplotlib.pyplot as plt
# plt.close('all')
# plt.figure(1)
# plt.plot(s, integrand_loc/np.max(np.abs(integrand_loc)), label='loc real')
# plt.plot(s, -integrand_rdt.real/np.max(np.abs(integrand_rdt)), label='rdt real')
# plt.plot(s, -integrand_rdt.imag/np.max(np.abs(integrand_rdt)), label='rdt imag')
# plt.plot(s, np.abs(integrand_rdt)/np.max(np.abs(integrand_rdt)), label='rdt abs')
# plt.legend()
# plt.title('Integrand for f0040')

# plt.show()
