import yaml
import json
import xobjects as xo

import xtrack as xt

# Read config file
with open('config_knobs_and_tuning.yaml','r') as fid:
    configuration = yaml.safe_load(fid)

# Load collider
with open('collider_01_bb_off.json', 'r') as fid:
    collider = xt.Multiline.from_dict(json.load(fid))

# Load orbit correction configuration
with open('corr_co.json', 'r') as fid:
    co_corr_config = json.load(fid)

# Set all knobs (crossing angles, dispersion correction, rf, crab cavities,
# experimental magnets, etc.)
for kk, vv in configuration['knob_settings'].items():
    collider.vars[kk] = vv

# Build trackers
collider.build_trackers()

# Twiss before correction
twb1_before = collider['lhcb1'].twiss()
twb2_before = collider['lhcb2'].twiss(reverse=True)

# Tunings
for line_name in ['lhcb1', 'lhcb2']:
    knob_names = configuration['knob_names'][line_name]

    # Correct closed orbit
    print(f'Correcting closed orbit for {line_name}')
    collider[line_name].correct_closed_orbit(
                            reference=collider[line_name+'_co_ref'],
                            correction_config=co_corr_config[line_name])

    # Match coupling
    print(f'Matching coupling for {line_name}')
    collider[line_name].match(
        vary=[
            xt.Vary(name=knob_names['c_minus_knob_1'],
                    limits=[-0.5e-2, 0.5e-2], step=1e-5),
            xt.Vary(name=knob_names['c_minus_knob_2'],
                    limits=[-0.5e-2, 0.5e-2], step=1e-5)],
        targets=[xt.Target('c_minus', 0, tol=1e-4)])

    # Match tune and chromaticity
    print(f'Matching tune and chromaticity for {line_name}')
    collider[line_name].match(verbose=False,
        vary=[
            xt.Vary(knob_names['q_knob_1'], step=1e-8),
            xt.Vary(knob_names['q_knob_2'], step=1e-8),
            xt.Vary(knob_names['dq_knob_1'], step=1e-4),
            xt.Vary(knob_names['dq_knob_2'], step=1e-4),
        ],
        targets = [
            xt.Target('qx', configuration['qx'][line_name], tol=1e-4),
            xt.Target('qy', configuration['qy'][line_name], tol=1e-4),
            xt.Target('dqx', configuration['dqx'][line_name], tol=0.05),
            xt.Target('dqy', configuration['dqy'][line_name], tol=0.05)])

# # Configure beam-beam lenses
# print('Configuring beam-beam lenses...')
# collider.configure_beambeam_interactions(
#     num_particles=2.2e11,
#     nemitt_x=2e-6, nemitt_y=3e-6)


with open('collider_02_bb_on.json', 'w') as fid:
    dct = collider.to_dict()
    json.dump(dct, fid, cls=xo.JEncoder)


# Checks
collider.vars['beambeam_scale'] = 0.0 # Switch off beam-beam
                                      # Beam-beam lenses are checked in separate script

import numpy as np
for line_name in ['lhcb1', 'lhcb2']:
    tw = collider[line_name].twiss()

    assert np.isclose(tw.qx, 62.31, atol=1e-5, rtol=0)
    assert np.isclose(tw.qy, 60.32, atol=1e-5, rtol=0)
    assert np.isclose(tw.qs, 0.00212, atol=1e-5, rtol=0) # Checks that RF is well set

    assert np.isclose(tw.dqx, 5, atol=0.1, rtol=0)
    assert np.isclose(tw.dqy, 6, atol=0.1, rtol=0)

    assert np.isclose(tw.c_minus, 0, atol=1e-4, rtol=0)
    assert np.allclose(tw.zeta, 0, rtol=0, atol=1e-4) # Check RF phase

    # Check separations
    assert np.isclose(tw['ip1', 'x'], 0, rtol=0, atol=5e-8) # sigma is 4e-6
    assert np.isclose(tw['ip1', 'y'], 0, rtol=0, atol=5e-8) # sigma is 4e-6
    assert np.isclose(tw['ip5', 'x'], 0, rtol=0, atol=5e-8) # sigma is 4e-6
    assert np.isclose(tw['ip5', 'y'], 0, rtol=0, atol=5e-8) # sigma is 4e-6

    assert np.isclose(tw['ip2', 'x'],
            -0.138e-3 * {'lhcb1': 1, 'lhcb2': 1}[line_name], # set separation
            rtol=0, atol=4e-6)
    assert np.isclose(tw['ip2', 'y'], 0, rtol=0, atol=5e-8)

    assert np.isclose(tw['ip8', 'x'], 0, rtol=0, atol=5e-8)
    assert np.isclose(tw['ip8', 'y'],
            -0.043e-3 * {'lhcb1': 1, 'lhcb2': -1}[line_name], # set separation
            rtol=0, atol=5e-8)

    # Check crossing angles
    assert np.isclose(tw['ip1', 'px'],
            250e-6 * {'lhcb1': 1, 'lhcb2': -1}[line_name], rtol=0, atol=0.5e-6)
    assert np.isclose(tw['ip1', 'py'], 0, rtol=0, atol=0.5e-6)
    assert np.isclose(tw['ip5', 'px'], 0, rtol=0, atol=0.5e-6)
    assert np.isclose(tw['ip5', 'py'], 250e-6, rtol=0, atol=0.5e-6)

    #assert np.isclose(tw['ip2', 'px'],




