import numpy as np
import matplotlib.pyplot as plt

import xtrack as xt


collider = xt.Multiline.from_json('./collider_03_tuned_bb_on.json')
collider.build_trackers()


collider.vars['beambeam_scale'] = 1
fp_ua = collider['lhcb1'].get_footprint(
    nemitt_x=2.5e-6, nemitt_y=2.5e-6,
    mode='uniform_action_grid',
    linear_rescale_on_knobs=[
        xt.LinearRescale(knob_name='beambeam_scale', v0=0.0, dv=0.1)]
    )

fp_polar = collider['lhcb1'].get_footprint(
    nemitt_x=2.5e-6, nemitt_y=2.5e-6,
    linear_rescale_on_knobs=[
        xt.LinearRescale(knob_name='beambeam_scale', v0=0.0, dv=0.1)]
    )

plt.close('all')

plt.figure(1)
fp_polar.plot()
plt.suptitle('Polar mode (default) - linear rescale on beambeam_scale')

plt.figure(2)
fp_ua.plot()
plt.suptitle('Uniform action grid mode - linear rescale on beambeam_scale')

plt.show()
