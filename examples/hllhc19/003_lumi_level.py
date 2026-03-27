from scipy.constants import c as clight

import xtrack as xt
import xmask as xm
import xmask.lhc as xlhc

# Read knobs and tuning settings from config file
with open('config.yaml','r') as fid:
    config = xm.yaml.load(fid)

lhc = xt.Environment.from_json(f'lhc_{config["label"]}_02_tuned_bb_off.json')

config_lumi_leveling = config['lumi_leveling']
config_beambeam = config['beam_beam']

opts = xlhc.luminosity_leveling(
    lhc, config_lumi_leveling=config_lumi_leveling,
    config_beambeam=config_beambeam)

# Re-match tunes, and chromaticities
conf_tuning = config['tuning']

for line_name in ['b1', 'b2']:
    knob_names = conf_tuning['knob_names'][line_name]
    targets = {
        'qx': conf_tuning['qx'][line_name],
        'qy': conf_tuning['qy'][line_name],
        'dqx': conf_tuning['dqx'][line_name],
        'dqy': conf_tuning['dqy'][line_name],
    }
    xm.machine_tuning(line=lhc[line_name],
        enable_tune_correction=True, enable_chromaticity_correction=True,
        knob_names=knob_names, targets=targets)

lhc.to_json(f'lhc_{config["label"]}_03_tuned_and_leveled_bb_off.json')
