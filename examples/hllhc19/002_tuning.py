import xtrack as xt
import xmask as xm

# Read knobs and tuning settings from config file
with open('config.yaml','r') as fid:
    config = xm.yaml.load(fid)

# Load collider anf build trackers
lhc = xt.load(f'lhc_{config["label"]}_01_multipolar_errors_corrected.json')

# Set all knobs (crossing angles, dispersion correction, rf, crab cavities,
# experimental magnets, etc.)
for kk, vv in config['knob_settings'].items():
    lhc[kk] = vv

# Reference model for orbit correction
env_ref = xt.load(f'lhc_co_ref_{config["label"]}.json')

# Tunings
conf_tuning = config['tuning']
optimizers = {}
for line_name in ['b1', 'b2']:
    print()
    print('Working on line ', line_name)

    knob_names = conf_tuning['knob_names'][line_name]

    targets = {
        'qx': conf_tuning['qx'][line_name],
        'qy': conf_tuning['qy'][line_name],
        'dqx': conf_tuning['dqx'][line_name],
        'dqy': conf_tuning['dqy'][line_name],
    }

    optimizers[line_name] = xm.machine_tuning(line=lhc[line_name],
        enable_closed_orbit_correction=True,
        enable_linear_coupling_correction=True,
        enable_tune_correction=True,
        enable_chromaticity_correction=True,
        knob_names=knob_names,
        targets=targets,
        step_q_knob=conf_tuning['steps']['q_knob'],
        step_dq_knob=conf_tuning['steps']['dq_knob'],
        step_c_minus_knob=conf_tuning['steps']['c_minus_knob'],
        tol_tune=conf_tuning['tolerances']['tune'],
        tol_chromaticity=conf_tuning['tolerances']['chromaticity'],
        tol_c_minus=conf_tuning['tolerances']['c_minus'],
        line_co_ref=env_ref[line_name],
        co_corr_config=conf_tuning['closed_orbit_correction'][line_name])

lhc.to_json(f'lhc_{config["label"]}_02_tuned_bb_off.json')
