import numpy as np

import xtrack as xt

collider = xt.Multiline.from_json('./collider_04_tuned_and_leveled_bb_on.json')
collider.build_trackers()

filling_pattern_cw = np.zeros(3564, dtype=int)
filling_pattern_acw = np.zeros(3564, dtype=int)

# harmonic_number = 35640
# bunch_spacing_buckets = 10

# ip_names = collider._bb_config['ip_names'] # is ['ip1', 'ip2', 'ip5', 'ip8']


# Some checks
dframes = collider._bb_config['dataframes']
assert (dframes['clockwise'].loc['bb_ho.c1b1_00', 'delay_in_slots'] == 0)
assert (dframes['clockwise'].loc['bb_ho.c5b1_00', 'delay_in_slots'] == 0)
assert (dframes['clockwise'].loc['bb_ho.c2b1_00', 'delay_in_slots'] == 891)
assert (dframes['clockwise'].loc['bb_ho.c8b1_00', 'delay_in_slots'] == 2670)

assert (dframes['anticlockwise'].loc['bb_ho.c1b2_00', 'delay_in_slots'] == 0)
assert (dframes['anticlockwise'].loc['bb_ho.c5b2_00', 'delay_in_slots'] == 0)
assert (dframes['anticlockwise'].loc['bb_ho.c2b2_00', 'delay_in_slots'] == 3564 - 891)
assert (dframes['anticlockwise'].loc['bb_ho.c8b2_00', 'delay_in_slots'] == 3564 - 2670)

assert (dframes['clockwise'].loc['bb_lr.r1b1_05', 'delay_in_slots'] == 0 + 5)
assert (dframes['clockwise'].loc['bb_lr.r5b1_05', 'delay_in_slots'] == 0 + 5)
assert (dframes['clockwise'].loc['bb_lr.r2b1_05', 'delay_in_slots'] == 891 + 5)
assert (dframes['clockwise'].loc['bb_lr.r8b1_05', 'delay_in_slots'] == 2670 + 5)

assert (dframes['anticlockwise'].loc['bb_lr.r1b2_05', 'delay_in_slots'] == 0 - 5)
assert (dframes['anticlockwise'].loc['bb_lr.r5b2_05', 'delay_in_slots'] == 0 - 5)
assert (dframes['anticlockwise'].loc['bb_lr.r2b2_05', 'delay_in_slots'] == 3564 - 891 - 5)
assert (dframes['anticlockwise'].loc['bb_lr.r8b2_05', 'delay_in_slots'] == 3564 - 2670 - 5)

assert (dframes['clockwise'].loc['bb_lr.l1b1_05', 'delay_in_slots'] == 0 - 5)
assert (dframes['clockwise'].loc['bb_lr.l5b1_05', 'delay_in_slots'] == 0 - 5)
assert (dframes['clockwise'].loc['bb_lr.l2b1_05', 'delay_in_slots'] == 891 - 5)
assert (dframes['clockwise'].loc['bb_lr.l8b1_05', 'delay_in_slots'] == 2670 - 5)

assert (dframes['anticlockwise'].loc['bb_lr.l1b2_05', 'delay_in_slots'] == 0 + 5)
assert (dframes['anticlockwise'].loc['bb_lr.l5b2_05', 'delay_in_slots'] == 0 + 5)
assert (dframes['anticlockwise'].loc['bb_lr.l2b2_05', 'delay_in_slots'] == 3564 - 891 + 5)
assert (dframes['anticlockwise'].loc['bb_lr.l8b2_05', 'delay_in_slots'] == 3564 - 2670 + 5)

# Apply filling scheme

# Single in bucket 0
filling_pattern_cw *= 0 # Reset
filling_pattern_acw *= 0 # Reset

filling_pattern_cw[1000] = 1
filling_pattern_acw[1000] = 1

i_bunch_cw = 1000
i_bunch_acw = 1000

collider.apply_filling_pattern(
    filling_pattern_cw=filling_pattern_cw,
    filling_pattern_acw=filling_pattern_acw,
    i_bunch_cw=i_bunch_cw, i_bunch_acw=i_bunch_acw)

twb1 = collider.lhcb1.twiss()
twb2 = collider.lhcb2.twiss()

# Check that only head-on lenses in ip1 and ip5 are enabled
all_bb_lenses_b1 = twb1.rows['bb_.*'].name
assert np.sum([collider.lhcb1[nn].scale_strength for nn in all_bb_lenses_b1]) == 22 # 11 in IP1 and 11 in IP5
all_bb_lenses_b2 = twb2.rows['bb_.*'].name
assert np.sum([collider.lhcb2[nn].scale_strength for nn in all_bb_lenses_b2]) == 22 # 11 in IP1 and 11 in IP5

assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*1b1.*'].name]) == 11 # 11 in IP1
assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*5b1.*'].name]) == 11 # 11 in IP5
assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*1b2.*'].name]) == 11 # 11 in IP1
assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*5b2.*'].name]) == 11 # 11 in IP5
assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*2b1.*'].name]) == 0 # 0 in IP2
assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*8b1.*'].name]) == 0 # 0 in IP8
assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*2b2.*'].name]) == 0 # 0 in IP2
assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*8b2.*'].name]) == 0 # 0 in IP8

assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_lr.*'].name]) == 0 # Long range
assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_lr.*'].name]) == 0 # Long range