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

assert np.isclose(tw.lhcb1['dqx'], 5, rtol=0, atol=0.01)
assert np.isclose(tw.lhcb1['dqy'], 6, rtol=0, atol=0.01)
assert np.isclose(tw.lhcb2['dqx'], 5, rtol=0, atol=0.01)
assert np.isclose(tw.lhcb2['dqy'], 6, rtol=0, atol=0.01)

# Check luminosity in ip8
ll_ip8 = xt.lumi.luminosity_from_twiss(
    n_colliding_bunches=2572,
    num_particles_per_bunch=2.2e11,
    ip_name='ip8',
    nemitt_x=2.5e-6,
    nemitt_y=2.5e-6,
    sigma_z=0.076,
    twiss_b1=tw.lhcb1,
    twiss_b2=tw.lhcb2,
    crab=False)

assert np.isclose(ll_ip8, 2e33, rtol=1e-2, atol=0)

# Check separation in ip2
mean_betx = np.sqrt(tw['lhcb1']['betx', 'ip2']
                 *tw['lhcb2']['betx', 'ip2'])
gamma0 = tw['lhcb1'].particle_on_co.gamma0[0]
beta0 = tw['lhcb1'].particle_on_co.beta0[0]
sigmax = np.sqrt(2.5e-6 * mean_betx /gamma0 / beta0)

assert np.isclose(collider.vars['on_sep2']._value/1000, 5*sigmax/2, rtol=1e-3, atol=0)