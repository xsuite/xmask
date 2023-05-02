import numpy as np

import xtrack as xt

collider = xt.Multiline.from_json('./collider_04_tuned_and_leveled_bb_on.json')
collider.build_trackers()

filling_pattern_b1 = np.zeros(3564, dtype=int)
filling_pattern_b2 = np.zeros(3564, dtype=int)

# Fill 50 bunches around bunch 500
filling_pattern_b1[500-25:500+25] = 1


ip_names = collider._bb_config['ip_names'] # is ['ip1', 'ip2', 'ip5', 'ip8']
delay_at_ips_slots = [0, 891, 0, 2670]

delay_at_ips_dict = {iipp: dd for iipp, dd in zip(ip_names, delay_at_ips_slots)}

# Start working on B1
bbdf = collider._bb_config['dataframes']['clockwise']

delay_in_slots = []

for nn in bbdf.index.values:
    ip_name = bbdf.loc[nn, 'ip_name']
    this_delay = delay_at_ips_dict[ip_name]

    if nn.startswith('bb_lr.'):
        this_delay += bbdf.loc[nn, 'identifier']

    delay_in_slots.append(this_delay)

bbdf['delay_in_slots'] = delay_in_slots



