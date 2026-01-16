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

# Attach reference particle (TODO: generalize for ions)
for bb in ['b1','b2']:
    lhc[bb].set_particle_ref('proton',
         energy0=config['beam_config'][bb]['beam_energy_tot'] * 1e9)

twb1 = lhc.b1.twiss4d()
twb2 = lhc.b2.twiss4d()





