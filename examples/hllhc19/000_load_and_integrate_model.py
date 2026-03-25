import xtrack as xt
import xmask as xm
import xmask.lhc as xlhc
from _temp_slice_lattice import slice_lattice

# Read config file
with open('config.yaml','r') as fid:
    config = xm.yaml.load(fid)

# Load lattice
lhc = xt.load(config['lattice_file'])

# Clear default twiss settings
lhc.b1.twiss_default.clear()
lhc.b2.twiss_default.clear()

# Load optics
lhc.vars.load(config['optics_file'])

# Create reference particles (TODO: generalize for ions)
lhc.new_particle(f'particle_ref_b1',
            energy0=config['beam_config']['b1']['beam_energy_tot'] * 1e9)
lhc.new_particle(f'particle_ref_b2',
            energy0=config['beam_config']['b2']['beam_energy_tot'] * 1e9)

# Assign reference particles to beams
lhc.b1.particle_ref = 'particle_ref_b1'
lhc.b2.particle_ref = 'particle_ref_b2'

# Define reference energy and rigidity variables
lhc['energy0_b1'] = lhc.ref['particle_ref_b1'].energy0[0]
lhc['energy0_b2'] = lhc.ref['particle_ref_b2'].energy0[0]
lhc['brho0_b1'] = lhc.ref['particle_ref_b1'].rigidity0[0]
lhc['brho0_b2'] = lhc.ref['particle_ref_b2'].rigidity0[0]

# Define new knobs from yaml
lhc.vars.default_to_zero = True # for knobs defined implicitly within expressions
for knob_name, knob_expr in config['new_knobs'].items():
    lhc[knob_name] = knob_expr
lhc.vars.default_to_zero = False

# Attach orbit correction knobs to all dipole correctors
lhc['on_corr_co'] = 1
for kk in list(lhc.vars.keys()):
    if kk.startswith('acb'):
        lhc['corr_co_'+kk] = 0
        lhc.ref[kk] += (lhc.ref['corr_co_'+kk] * lhc.ref['on_corr_co'])

# Cycle both beams
lhc.b1.cycle('ip3')
lhc.b2.cycle('ip3')

# Install beam-beam lenses (inactive and not configured)
config_bb = config['beam_beam']
if config_bb['install_beam_beam']:
    lhc.install_beambeam_interactions(
        clockwise_line='b1',
        anticlockwise_line='b2',
        ip_names=['ip1', 'ip2', 'ip5', 'ip8'],
        delay_at_ips_slots=[0, 891, 0, 2670],
        num_long_range_encounters_per_side=
            config_bb['num_long_range_encounters_per_side'],
        num_slices_head_on=config_bb['num_slices_head_on'],
        harmonic_number=35640,
        bunch_spacing_buckets=config_bb['bunch_spacing_buckets'],
        sigmaz=config_bb['sigma_z'])

# Prepare reference model for orbit correction
lhc_co_ref = xlhc.build_closed_orbit_reference(lhc)
lhc_co_ref.to_json(f'lhc_co_ref_{config["label"]}.json')

lhc.to_json(f'lhc_{config["label"]}_00_prepared.json')

# Check that both lines twiss without errors
twb1 = lhc.b1.twiss4d()
twb2 = lhc.b2.twiss4d()
