from scipy.constants import c as clight

import xtrack as xt
import xmask as xm

# Load collider anf build trackers
collider = xt.Multiline.from_json('collider_02_tuned_bb_off.json')
collider.build_trackers()

# Read knobs and tuning settings from config file
with open('config.yaml','r') as fid:
    config = xm.yaml.load(fid)

config_lumi_leveling_ip2_ip8 = config['config_lumi_leveling_ip2_ip8']
config_beambeam = config['config_beambeam']

# Leveling in IP2
print('\n --- Leveling in IP2 ---')
config_ip2 = config_lumi_leveling_ip2_ip8['ip2']
collider.match(
    lines=['lhcb1', 'lhcb2'],
    ele_start=['e.ds.l2.b1', 's.ds.r2.b2'],
    ele_stop=['s.ds.r2.b1', 'e.ds.l2.b2'],
    twiss_init='preserve',
    targets=[
        xt.TargetSeparation(
            ip_name='ip2', separation_norm=config_ip2['separation_in_sigmas'],
            tol=1e-4, # in sigmas
            plane=config_ip2['plane'],
            nemitt_x=config_beambeam['nemitt_x'],
            nemitt_y=config_beambeam['nemitt_y']),
        # Preserve crossing angle
        xt.TargetList(['px', 'py'], at='ip2', line='lhcb1', value='preserve', tol=1e-7, scale=1e3),
        xt.TargetList(['px', 'py'], at='ip2', line='lhcb2', value='preserve', tol=1e-7, scale=1e3),
        # Close the bumps
        xt.TargetList(['x', 'y'], at='s.ds.r2.b1', line='lhcb1', value='preserve', tol=1e-5, scale=1),
        xt.TargetList(['px', 'py'], at='s.ds.r2.b1', line='lhcb1', value='preserve', tol=1e-5, scale=1e3),
        xt.TargetList(['x', 'y'], at='e.ds.l2.b2', line='lhcb2', value='preserve', tol=1e-5, scale=1),
        xt.TargetList(['px', 'py'], at='e.ds.l2.b2', line='lhcb2', value='preserve', tol=1e-5, scale=1e3),
    ],
    vary=
        [xt.Vary('on_sep2', step=1e-4),
         xt.VaryList([
            # correctors to control the crossing angles
            'corr_co_acbyvs4.l2b1', 'corr_co_acbyhs4.l2b1',
            'corr_co_acbyvs4.r2b2', 'corr_co_acbyhs4.r2b2',
             # correctors to close the bumps
            'corr_co_acbyvs4.l2b2', 'corr_co_acbyhs4.l2b2',
            'corr_co_acbyvs4.r2b1', 'corr_co_acbyhs4.r2b1',
            'corr_co_acbyhs5.l2b2', 'corr_co_acbyvs5.l2b2',
            'corr_co_acbchs5.r2b1', 'corr_co_acbcvs5.r2b1'],
            step=1e-7),
        ],
)

# Leveling in IP8
print('\n --- Leveling in IP8 ---')
config_ip8 = config_lumi_leveling_ip2_ip8['ip8']

beta0_b1 = collider.lhcb1.particle_ref.beta0[0]
f_rev=1/(collider.lhcb1.get_length() /(beta0_b1 * clight))

targets_ip8=[]
vary_ip8=[]

targets_ip8.append(
    xt.TargetLuminosity(
        ip_name='ip8', luminosity=config_ip8['luminosity'], crab=False,
        tol=0.01 * config_ip8['luminosity'],
        f_rev=f_rev, num_colliding_bunches=config_ip8['num_colliding_bunches'],
        num_particles_per_bunch=config_beambeam['num_particles_per_bunch'],
        sigma_z=config_beambeam['sigma_z'],
        nemitt_x=config_beambeam['nemitt_x'],
        nemitt_y=config_beambeam['nemitt_y'])
)

if config_ip8['impose_separation_orthogonal_to_crossing']:
    targets_ip8.append(
        xt.TargetSeparationOrthogonalToCrossing(ip_name='ip8'))
    vary_ip8.append(
        xt.VaryList(['on_sep8h', 'on_sep8v'], step=1e-4)) # to control separation in x and y
else:
    vary_ip8.append(
        xt.Vary('on_sep8', step=1e-4)) # to control separation in the separation plane defined by the optics

# Target and knobs to rematch the crossing angles and close the bumps
targets_ip8 += [
    # Preserve crossing angle
    xt.TargetList(['px', 'py'], at='ip8', line='lhcb1', value='preserve', tol=1e-7, scale=1e3),
    xt.TargetList(['px', 'py'], at='ip8', line='lhcb2', value='preserve', tol=1e-7, scale=1e3),
    # Close the bumps
    xt.TargetList(['x', 'y'], at='s.ds.r8.b1', line='lhcb1', value='preserve', tol=1e-5, scale=1),
    xt.TargetList(['px', 'py'], at='s.ds.r8.b1', line='lhcb1', value='preserve', tol=1e-5, scale=1e3),
    xt.TargetList(['x', 'y'], at='e.ds.l8.b2', line='lhcb2', value='preserve', tol=1e-5, scale=1),
    xt.TargetList(['px', 'py'], at='e.ds.l8.b2', line='lhcb2', value='preserve', tol=1e-5, scale=1e3),
    ]
vary_ip8 += [
        xt.VaryList([
            # correctors to control the crossing angles
            'corr_co_acbyvs4.l8b1', 'corr_co_acbyhs4.l8b1',
            'corr_co_acbyvs4.r8b2', 'corr_co_acbyhs4.r8b2',
             # correctors to close the bumps
            'corr_co_acbyvs4.l8b2', 'corr_co_acbyhs4.l8b2',
            'corr_co_acbyvs4.r8b1', 'corr_co_acbyhs4.r8b1',
            'corr_co_acbcvs5.l8b2', 'corr_co_acbchs5.l8b2',
            'corr_co_acbyvs5.r8b1', 'corr_co_acbyhs5.r8b1'],
            step=1e-7),
    ]

# Leveling with crossing angle and bump rematching
collider.match(
    lines=['lhcb1', 'lhcb2'],
    ele_start=['e.ds.l8.b1', 's.ds.r8.b2'],
    ele_stop=['s.ds.r8.b1', 'e.ds.l8.b2'],
    twiss_init='preserve',
    targets=targets_ip8,
    vary=vary_ip8
)

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