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

twb1 = collider.lhcb1.twiss()
twb2 = collider.lhcb2.twiss()

##############################################
# Check with only one head-on in IP1 and IP5 #
##############################################

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


######################################
# Check with only one head-on in IP8 #
######################################

filling_pattern_cw *= 0 # Reset
filling_pattern_acw *= 0 # Reset

# These are supposed to collide in IP8 (checked with LPC tool)
filling_pattern_cw[174] = 1
filling_pattern_acw[2844] = 1

i_bunch_cw = 174
i_bunch_acw = 2844

collider.apply_filling_pattern(
    filling_pattern_cw=filling_pattern_cw,
    filling_pattern_acw=filling_pattern_acw,
    i_bunch_cw=i_bunch_cw, i_bunch_acw=i_bunch_acw)

all_bb_lenses_b1 = twb1.rows['bb_.*'].name
assert np.sum([collider.lhcb1[nn].scale_strength for nn in all_bb_lenses_b1]) == 11 # 11 in IP8
all_bb_lenses_b2 = twb2.rows['bb_.*'].name
assert np.sum([collider.lhcb2[nn].scale_strength for nn in all_bb_lenses_b2]) == 11 # 11 in IP8

assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*1b1.*'].name]) == 0 # IP1
assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*5b1.*'].name]) == 0 # IP5
assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*1b2.*'].name]) == 0 # IP1
assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*5b2.*'].name]) == 0 # IP5
assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*2b1.*'].name]) == 0 # IP2
assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*8b1.*'].name]) == 11 # IP8
assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*2b2.*'].name]) == 0  # IP2
assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*8b2.*'].name]) == 11 # IP8

assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_lr.*'].name]) == 0 # Long range
assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_lr.*'].name]) == 0 # Long range


######################################
# Check with only one head-on in IP2 #
######################################

filling_pattern_cw *= 0 # Reset
filling_pattern_acw *= 0 # Reset

# These are supposed to collide in IP2 (checked with LPC tool)
filling_pattern_cw[2952] = 1
filling_pattern_acw[279] = 1

i_bunch_cw = 2952
i_bunch_acw = 279

collider.apply_filling_pattern(
    filling_pattern_cw=filling_pattern_cw,
    filling_pattern_acw=filling_pattern_acw,
    i_bunch_cw=i_bunch_cw, i_bunch_acw=i_bunch_acw)

all_bb_lenses_b1 = twb1.rows['bb_.*'].name
assert np.sum([collider.lhcb1[nn].scale_strength for nn in all_bb_lenses_b1]) == 11 # 11 in IP2
all_bb_lenses_b2 = twb2.rows['bb_.*'].name
assert np.sum([collider.lhcb2[nn].scale_strength for nn in all_bb_lenses_b2]) == 11 # 11 in IP2

assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*1b1.*'].name]) == 0 # IP1
assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*5b1.*'].name]) == 0 # IP5
assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*1b2.*'].name]) == 0 # IP1
assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*5b2.*'].name]) == 0 # IP5
assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*2b1.*'].name]) == 11 # IP2
assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*8b1.*'].name]) == 0  # IP8
assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*2b2.*'].name]) == 11 # IP2
assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*8b2.*'].name]) == 0  # IP8

assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_lr.*'].name]) == 0 # Long range
assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_lr.*'].name]) == 0 # Long range

########################################################
# Check with one long range on the left of IP1 and IP5 #
########################################################

filling_pattern_cw *= 0 # Reset
filling_pattern_acw *= 0 # Reset
filling_pattern_cw[1000 + 5] = 1
filling_pattern_acw[1000] = 1

i_bunch_cw = 1000 + 5 # Long range expected on the left
i_bunch_acw = 1000

collider.apply_filling_pattern(
    filling_pattern_cw=filling_pattern_cw,
    filling_pattern_acw=filling_pattern_acw,
    i_bunch_cw=i_bunch_cw, i_bunch_acw=i_bunch_acw)


# Check that only head-on lenses in ip1 and ip5 are enabled
all_bb_lenses_b1 = twb1.rows['bb_.*'].name
assert np.sum([collider.lhcb1[nn].scale_strength for nn in all_bb_lenses_b1]) == 2 # one long range in each of the main ips
all_bb_lenses_b2 = twb2.rows['bb_.*'].name
assert np.sum([collider.lhcb2[nn].scale_strength for nn in all_bb_lenses_b2]) == 2 # one long range in each of the main ips

assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*1b1.*'].name]) == 0 # IP1
assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*5b1.*'].name]) == 0 # IP5
assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*1b2.*'].name]) == 0 # IP1
assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*5b2.*'].name]) == 0 # IP5
assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*2b1.*'].name]) == 0 # IP2
assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*8b1.*'].name]) == 0 # IP8
assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*2b2.*'].name]) == 0 # IP2
assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*8b2.*'].name]) == 0 # IP8

assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_lr.*'].name]) == 2 # Long range
assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_lr.*'].name]) == 2 # Long range

assert collider.lhcb1['bb_lr.l5b1_05'].scale_strength == 1
assert collider.lhcb1['bb_lr.l1b1_05'].scale_strength == 1
assert collider.lhcb2['bb_lr.l5b2_05'].scale_strength == 1
assert collider.lhcb2['bb_lr.l1b2_05'].scale_strength == 1

#################################################
# Check with one long range on the right of IP2 #
#################################################

filling_pattern_cw *= 0 # Reset
filling_pattern_acw *= 0 # Reset

# These are supposed to collide in IP2 (checked with LPC tool)
filling_pattern_cw[2952] = 1
filling_pattern_acw[279 + 5] = 1

i_bunch_cw = 2952
i_bunch_acw = 279 + 5

collider.apply_filling_pattern(
    filling_pattern_cw=filling_pattern_cw,
    filling_pattern_acw=filling_pattern_acw,
    i_bunch_cw=i_bunch_cw, i_bunch_acw=i_bunch_acw)

all_bb_lenses_b1 = twb1.rows['bb_.*'].name
assert np.sum([collider.lhcb1[nn].scale_strength for nn in all_bb_lenses_b1]) == 1 # IP2
all_bb_lenses_b2 = twb2.rows['bb_.*'].name
assert np.sum([collider.lhcb2[nn].scale_strength for nn in all_bb_lenses_b2]) == 1 # IP2

assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*1b1.*'].name]) == 0 # IP1
assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*5b1.*'].name]) == 0 # IP5
assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*1b2.*'].name]) == 0 # IP1
assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*5b2.*'].name]) == 0 # IP5
assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*2b1.*'].name]) == 0 # IP2
assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*8b1.*'].name]) == 0 # IP8
assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*2b2.*'].name]) == 0 # IP2
assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*8b2.*'].name]) == 0 # IP8

assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_lr.*'].name]) == 1 # Long range
assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_lr.*'].name]) == 1 # Long range

assert collider.lhcb1['bb_lr.r2b1_05'].scale_strength == 1
assert collider.lhcb2['bb_lr.r2b2_05'].scale_strength == 1

#####################################
# Many long ranges only on one side #
#####################################

filling_pattern_cw *= 0 # Reset
filling_pattern_acw *= 0 # Reset

filling_pattern_cw[1565 : 1565 + 48] = 1
filling_pattern_acw[718 : 718 + 48] = 1
filling_pattern_acw[1612 : 1612 + 48] = 1

i_bunch_cw = 1612
i_bunch_acw = 1612

collider.apply_filling_pattern(
    filling_pattern_cw=filling_pattern_cw,
    filling_pattern_acw=filling_pattern_acw,
    i_bunch_cw=i_bunch_cw, i_bunch_acw=i_bunch_acw)

all_bb_lenses_b1 = twb1.rows['bb_.*'].name
assert np.sum([collider.lhcb1[nn].scale_strength for nn in all_bb_lenses_b1]) == (
      11 # head on IP1
    + 25 # long-range on one side of IP1
    + 11 # head on IP5
    + 25 # long-range on one side of IP5
    + 11 # head on IP2
    + 20 # long-range on one side of IP8
)
all_bb_lenses_b2 = twb2.rows['bb_.*'].name
assert np.sum([collider.lhcb2[nn].scale_strength for nn in all_bb_lenses_b2]) == (
      11 # head on IP1
    + 25 # long-range on one side of IP1
    + 11 # head on IP5
    + 25 # long-range on one side of IP5
)

assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*1b1.*'].name]) == 11  # IP1
assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*5b1.*'].name]) == 11  # IP5
assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*1b2.*'].name]) == 11  # IP1
assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*5b2.*'].name]) == 11  # IP5
assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*2b1.*'].name]) == 0   # IP2
assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*8b1.*'].name]) == 11  # IP8
assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*2b2.*'].name]) == 0   # IP2
assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*8b2.*'].name]) == 0   # IP8

assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_lr.r1b1_.*'].name]) == 25
assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_lr.l1b1_.*'].name]) == 0
assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_lr.r1b2_.*'].name]) == 25
assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_lr.l1b2_.*'].name]) == 0

assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_lr.r5b1_.*'].name]) == 25
assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_lr.l5b1_.*'].name]) == 0
assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_lr.r5b2_.*'].name]) == 25
assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_lr.l5b2_.*'].name]) == 0

assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_lr.r2b1_.*'].name]) == 0
assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_lr.l2b1_.*'].name]) == 0
assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_lr.r2b2_.*'].name]) == 0
assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_lr.l2b2_.*'].name]) == 0

assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_lr.r8b1_.*'].name]) == 20
assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_lr.l8b1_.*'].name]) == 0
assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_lr.r8b2_.*'].name]) == 0
assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_lr.l8b2_.*'].name]) == 0

###############################################
# A case where all bb lenses should be active #
###############################################

filling_pattern_cw *= 0 # Reset
filling_pattern_acw *= 0 # Reset

filling_pattern_cw[881 : 881 + 72] = 1
filling_pattern_cw[1775 : 1775 + 72] = 1
filling_pattern_cw[2669 : 2669 + 72] = 1

filling_pattern_acw[881 : 881 + 72] = 1
filling_pattern_acw[1775 : 1775 + 72] = 1
filling_pattern_acw[2669 : 2669 + 72] = 1

i_bunch_cw = 1775 + 36
i_bunch_acw = 1775 + 36

collider.apply_filling_pattern(
    filling_pattern_cw=filling_pattern_cw,
    filling_pattern_acw=filling_pattern_acw,
    i_bunch_cw=i_bunch_cw, i_bunch_acw=i_bunch_acw)

all_bb_lenses_b1 = twb1.rows['bb_.*'].name
assert np.all(np.array([collider.lhcb1[nn].scale_strength for nn in all_bb_lenses_b1]) == 1)
all_bb_lenses_b2 = twb2.rows['bb_.*'].name
assert np.all(np.array([collider.lhcb2[nn].scale_strength for nn in all_bb_lenses_b2]) == 1)
