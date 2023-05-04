from scipy.constants import c as clight

import xtrack as xt
import xmask as xm
import xmask.lhc as xlhc

# Load collider anf build trackers
collider = xt.Multiline.from_json('collider_02_tuned_bb_off.json')
collider.build_trackers()

# Read knobs and tuning settings from config file
with open('config.yaml','r') as fid:
    config = xm.yaml.load(fid)

config_lumi_leveling = config['config_lumi_leveling']
config_beambeam = config['config_beambeam']

xlhc.luminosity_leveling(
    collider, config_lumi_leveling=config_lumi_leveling,
    config_beambeam=config_beambeam)

# Re-match tunes, and chromaticities
conf_knobs_and_tuning = config['config_knobs_and_tuning']

for line_name in ['lhcb1', 'lhcb2']:
    knob_names = conf_knobs_and_tuning['knob_names'][line_name]
    targets = {
        'qx': conf_knobs_and_tuning['qx'][line_name],
        'qy': conf_knobs_and_tuning['qy'][line_name],
        'dqx': conf_knobs_and_tuning['dqx'][line_name],
        'dqy': conf_knobs_and_tuning['dqy'][line_name],
    }
    xm.machine_tuning(line=collider[line_name],
        enable_tune_correction=True, enable_chromaticity_correction=True,
        knob_names=knob_names, targets=targets)

collider.to_json('collider_03_tuned_and_leveled_bb_off.json')




# Checks
import numpy as np
tw = collider.twiss(lines=['lhcb1', 'lhcb2'])

assert np.isclose(tw.lhcb1['qx'], 62.31, rtol=0, atol=1e-5)
assert np.isclose(tw.lhcb1['qy'], 60.32, rtol=0, atol=1e-5)
assert np.isclose(tw.lhcb2['qx'], 62.31, rtol=0, atol=1e-5)
assert np.isclose(tw.lhcb2['qy'], 60.32, rtol=0, atol=1e-5)

assert np.isclose(tw.lhcb1['dqx'], 10, rtol=0, atol=0.03)
assert np.isclose(tw.lhcb1['dqy'], 10, rtol=0, atol=0.03)
assert np.isclose(tw.lhcb2['dqx'], 10, rtol=0, atol=0.03)
assert np.isclose(tw.lhcb2['dqy'], 10, rtol=0, atol=0.03)

# Check luminosity in ip8
ll_ip8 = xt.lumi.luminosity_from_twiss(
    n_colliding_bunches=398,
    num_particles_per_bunch=180000000.,
    ip_name='ip8',
    nemitt_x=1.65e-6,
    nemitt_y=1.65e-6,
    sigma_z=0.0824,
    twiss_b1=tw.lhcb1,
    twiss_b2=tw.lhcb2,
    crab=False)

assert np.isclose(ll_ip8, 1.0e+27 , rtol=1e-2, atol=0)

# Check luminosity in ip2
ll_ip2 = xt.lumi.luminosity_from_twiss(
    n_colliding_bunches=1088,
    num_particles_per_bunch=180000000.,
    ip_name='ip2',
    nemitt_x=1.65e-6,
    nemitt_y=1.65e-6,
    sigma_z=0.0824,
    twiss_b1=tw.lhcb1,
    twiss_b2=tw.lhcb2,
    crab=False)

assert np.isclose(ll_ip2, 6.4e+27 , rtol=1e-2, atol=0)

# Check luminosity in ip1
ll_ip1 = xt.lumi.luminosity_from_twiss(
    n_colliding_bunches=1088,
    num_particles_per_bunch=180000000.,
    ip_name='ip1',
    nemitt_x=1.65e-6,
    nemitt_y=1.65e-6,
    sigma_z=0.0824,
    twiss_b1=tw.lhcb1,
    twiss_b2=tw.lhcb2,
    crab=False)

assert np.isclose(ll_ip1, 6.4e+27 , rtol=1e-2, atol=0)

# Check luminosity in ip5
ll_ip5 = xt.lumi.luminosity_from_twiss(
    n_colliding_bunches=1088,
    num_particles_per_bunch=180000000.,
    ip_name='ip1',
    nemitt_x=1.65e-6,
    nemitt_y=1.65e-6,
    sigma_z=0.0824,
    twiss_b1=tw.lhcb1,
    twiss_b2=tw.lhcb2,
    crab=False)

assert np.isclose(ll_ip5, 6.4e+27 , rtol=1e-2, atol=0)