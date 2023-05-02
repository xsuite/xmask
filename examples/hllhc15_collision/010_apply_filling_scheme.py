import numpy as np

import xtrack as xt

collider = xt.Multiline.from_json('./collider_04_tuned_and_leveled_bb_on.json')
collider.build_trackers()

filling_pattern_cw = np.zeros(3564, dtype=int)
filling_pattern_acw = np.zeros(3564, dtype=int)

harmonic_number = 35640
bunch_spacing_buckets = 10

ip_names = collider._bb_config['ip_names'] # is ['ip1', 'ip2', 'ip5', 'ip8']
delay_at_ips_slots = [0, 891, 0, 2670] # Defined as anticlockwise bunch id that
                                       # meets bunch 0 of the clockwise beam

ring_length_in_slots = harmonic_number / bunch_spacing_buckets

for orientation in ['clockwise', 'anticlockwise']:

    if orientation == 'clockwise':
        delay_at_ips_dict = {iipp: dd
                             for iipp, dd in zip(ip_names, delay_at_ips_slots)}
    elif orientation == 'anticlockwise':
        delay_at_ips_dict = {iipp: np.mod(ring_length_in_slots - dd, ring_length_in_slots)
                             for iipp, dd in zip(ip_names, delay_at_ips_slots)}
    else:
        raise ValueError('?!')

    bbdf = collider._bb_config['dataframes'][orientation]

    delay_in_slots = []

    for nn in bbdf.index.values:
        ip_name = bbdf.loc[nn, 'ip_name']
        this_delay = delay_at_ips_dict[ip_name]

        if nn.startswith('bb_lr.'):
            if orientation == 'clockwise':
                this_delay += bbdf.loc[nn, 'identifier']
            elif orientation == 'anticlockwise':
                this_delay -= bbdf.loc[nn, 'identifier']
            else:
                raise ValueError('?!')

        delay_in_slots.append(int(this_delay))

    bbdf['delay_in_slots'] = delay_in_slots


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

# Work on clockwise
for orientation_self in ['clockwise', 'anticlockwise']:
    line_name_self = collider._bb_config[orientation_self + '_line']
    line_self = collider[line_name_self]

    if orientation_self == 'clockwise':
        filling_pattern_self = np.array(filling_pattern_cw, dtype=int)
        filling_pattern_other = np.array(filling_pattern_acw, dtype=int)
        i_bunch_self = i_bunch_cw
    else:
        filling_pattern_self = np.array(filling_pattern_acw)
        filling_pattern_other = np.array(filling_pattern_cw)
        i_bunch_self = i_bunch_acw

    assert set(list(filling_pattern_self)).issubset({0, 1})
    assert set(list(filling_pattern_other)).issubset({0, 1})

    assert filling_pattern_cw[i_bunch_self] == 1, "Selected bunch is not in the filling scheme"

    temp_df = dframes[orientation_self].loc[:, ['delay_in_slots', 'ip_name']].copy()
    temp_df['partner_bunch_index'] = dframes[orientation_self]['delay_in_slots'] + i_bunch_self
    temp_df['is_active'] = filling_pattern_other[temp_df['partner_bunch_index']] == 1

    for nn, state in temp_df['is_active'].items():
        if state:
            collider.vars[nn + '_scale_strength'] = collider.vars['beambeam_scale']
        else:
            collider.vars[nn + '_scale_strength'] = 0

    # Check that only the right lenses are activated
