import numpy as np

import xtrack as xt


collider = xt.Multiline.from_json('./collider_03_tuned_bb_on.json')
collider.build_trackers()

collider.vars['beambeam_scale'] = 0.3

fp = collider['lhcb1'].get_footprint(nemitt_x=2.5e-6, nemitt_y=2.5e-6,
                                     theta_range=(0.05, np.pi/2-0.05),
                                     n_fft=2**18)
# Find problematic points

i_problems = np.where(np.abs(fp.qx.flatten() - fp.qy.flatten()) < 1e-4)

fftp_x = np.squeeze(fp.fft_x[98, :])
fftp_y = np.squeeze(fp.fft_y[98, :])


fftp_y_scaled = np.abs(fftp_y) / np.max(np.abs(fftp_y)) * np.max(np.abs(fftp_x))

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
freq = np.fft.rfftfreq(len(fftp_x)*2 - 1)
plt.plot(freq, np.abs(fftp_x))
plt.plot(freq, fftp_y_scaled)
plt.plot(freq, np.abs(fftp_x) - np.abs(fftp_y_scaled))

plt.show()


# fpj = collider['lhcb1'].get_footprint(nemitt_x=1e-6, nemitt_y=1e-6,
#                                       mode='uniform_action_grid',
#                                       x_norm_range=(0.1, 6),
#                                       y_norm_range=(0.1, 6))
