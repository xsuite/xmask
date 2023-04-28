import numpy as np

import xtrack as xt

num_particles = 1e11
nemitt_x = 2.5e-6
nemitt_y = 2.5e-6

collider = xt.Multiline.from_json('collider_00_from_mad.json')
collider.build_trackers()

# Switch off the crab cavities
collider.vars['on_crab1'] = 0
collider.vars['on_crab5'] = 0


# Check that orbit is flat
tw = collider.twiss(method='4d')
assert np.max(np.abs(tw.lhcb1.x))< 1e-7

# Install head-on only
collider.discard_trackers()

collider.install_beambeam_interactions(
    clockwise_line='lhcb1',
    anticlockwise_line='lhcb2',
    ip_names=['ip1'],
    num_long_range_encounters_per_side=[0],
    num_slices_head_on=1,
    harmonic_number=35640,
    bunch_spacing_buckets=10,
    sigmaz=1e-6
)

collider.build_trackers()
# Switch on RF (assumes 6d)
collider.vars['vrf400'] = 16

collider.configure_beambeam_interactions(
    num_particles=num_particles,
    nemitt_x=nemitt_x, nemitt_y=nemitt_y,
    crab_strong_beam=False
)

fp = collider.lhcb1.get_footprint(nemitt_x=nemitt_x, nemitt_y=nemitt_y,)


