import numpy as np

import xtrack as xt


collider = xt.Multiline.from_json('./collider_03_tuned_bb_on.json')
collider.build_trackers()

# collider.vars['beambeam_scale'] = 0.0

# fp0 = collider['lhcb1'].get_footprint(nemitt_x=2.5e-6, nemitt_y=2.5e-6,
#                                      theta_range=(0.05, np.pi/2-0.05),
#                                      n_fft=2**18)


# collider.vars['beambeam_scale'] = 0.1

# fp1 = collider['lhcb1'].get_footprint(nemitt_x=2.5e-6, nemitt_y=2.5e-6,
#                                      theta_range=(0.05, np.pi/2-0.05),
#                                      n_fft=2**18)

# qx = fp0.qx + (fp1.qx - fp0.qx)/0.1*1.
# qy = fp0.qy + (fp1.qy - fp0.qy)/0.1*1.

from xtrack.footprint import _footprint_with_linear_rescale, LinearRescale

collider.vars['beambeam_scale'] = 1
fp_lr  = _footprint_with_linear_rescale(line= collider['lhcb1'],
        kwargs={'nemitt_x': 2.5e-6, 'nemitt_y': 2.5e-6,
                'theta_range': (0.05, np.pi/2-0.05),
                'n_fft': 2**18},
        linear_rescale_on_knobs=[
            LinearRescale(knob_name='beambeam_scale', v0=0.0, dv=0.1)])


