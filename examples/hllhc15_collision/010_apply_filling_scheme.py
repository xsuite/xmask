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

filling_pattern_cw[0] = 1
filling_pattern_acw[0] = 1

i_bunch_cw = 0
i_bunch_acw = 0

collider.apply_filling_pattern(
    filling_pattern_cw=filling_pattern_cw,
    filling_pattern_acw=filling_pattern_acw,
    i_bunch_cw=i_bunch_cw, i_bunch_acw=i_bunch_acw)
