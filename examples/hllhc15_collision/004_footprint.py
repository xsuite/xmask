import numpy as np
import matplotlib.pyplot as plt

import xtrack as xt


collider = xt.Multiline.from_json('./collider_03_tuned_bb_on.json')
collider.build_trackers()

fp0 = collider['lhcb1'].get_footprint(nemitt_x=2.5e-6, nemitt_y=2.5e-6)

fp_polar = collider['lhcb1'].get_footprint(
    nemitt_x=2.5e-6, nemitt_y=2.5e-6,
    linear_rescale_on_knobs=[
        xt.LinearRescale(knob_name='beambeam_scale', v0=0.0, dv=0.1)]
    )


fp_ua = collider['lhcb1'].get_footprint(
    nemitt_x=2.5e-6, nemitt_y=2.5e-6,
    mode='uniform_action_grid',
    linear_rescale_on_knobs=[
        xt.LinearRescale(knob_name='beambeam_scale', v0=0.0, dv=0.1)]
    )


plt.close('all')

fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
fp0.plot(ax=ax1, label='no rescale bb')
plt.suptitle('Polar mode (default) - no rescale on beambeam')

fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111, sharex=ax1, sharey=ax1)
fp_polar.plot(ax=ax2, label='rescale bb')
plt.suptitle('Polar mode (default) - linear rescale on beambeam')

fig3 = plt.figure(3)
ax3 = fig3.add_subplot(111, sharex=ax1, sharey=ax1)
fp_ua.plot()
plt.suptitle('Uniform action grid mode - linear rescale on beambeam')

plt.show()
