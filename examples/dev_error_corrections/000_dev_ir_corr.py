import xtrack as xt
from integral_correction import IntegralCorrection

env = xt.load('collider_00_from_mad_with_errors.json')

# Reference twiss
env_no_err = xt.load('collider_00_from_mad_no_errors.json')
tw = env_no_err.lhcb1.twiss4d() # Reference twiss


# Normal sextupole correction
correction_knobs=['kcsx3.l5', 'kcsx3.r5']
multipole='k2l'
target_quantities={'c12': (1, 2, 'diff'), 'c21': (2, 1, 'diff')}

# Normal octupole correction
# correction_knobs=['kcox3.l5', 'kcox3.r5']
# multipole='k3l'
# target_quantities={'c04': (0, 4, 'sum'), 'c40': (4, 0, 'sum')}
# # target_quantities={'f4000': 'f4000', 'f0040': 'f0040'}

# Usage:
rdt_contrib = IntegralCorrection(
                         line=env['lhcb1'],
                         tw=tw,
                         start='dfxj.4l5',
                         end='dfxj.4r5',
                         correction_knobs=correction_knobs,
                         multipole=multipole,
                         ip='ip5',
                         target_quantities=target_quantities,
                         generated_knob_name='on_corr_k3_ip5')
print("Original correction:")
rdt_contrib.print_corrections()

rdt_contrib.clear_corrections()
opt = rdt_contrib.correct()

print("Before setting the knob:")
rdt_contrib.print_corrections()

env[opt.knob_name] = 1.0
print("After setting the knob:")
rdt_contrib.print_corrections()

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
