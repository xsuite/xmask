import xtrack as xt
import xmask as xm

# Read config file
with open('config.yaml','r') as fid:
    config = xm.yaml.load(fid)

# Load lattice
lhc = xt.load('lhc.json')

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

# Define reference rigidity variables
lhc['brho0_b1'] = lhc.ref['particle_ref_b1'].rigidity0[0]
lhc['brho0_b2'] = lhc.ref['particle_ref_b2'].rigidity0[0]

# Define new knobs
lhc.vars.default_to_zero = True # for knobs defined implicitly within expressions
for knob_name, knob_expr in config['new_knobs'].items():
    lhc[knob_name] = knob_expr
lhc.vars.default_to_zero = False

# Define experimental magnet knobs

# Attach orbit correction knobs to all dipole correctors

# Prepare reference model for orbit correction


twb1 = lhc.b1.twiss4d()
twb2 = lhc.b2.twiss4d()








