import xtrack as xt
import xmask as xm
import xmask.lhc as xmlhc

env = xt.load('lhc_multipolar_errors.json')

# Read beam-beam config from config file
with open('config.yaml','r') as fid:
    config = xm.yaml.load(fid)

# Status of error knobs
tt_err_knobs = env.vars.get_table().rows[r'on_error_.*']
print("Error knobs in the environment:")
tt_err_knobs.show()

# Errors off to get reference twiss
env.set(tt_err_knobs.name, 0)
tw_b1 = env['lhcb1'].twiss4d(reverse=False) # Reference twiss
tw_b2 = env['lhcb2'].twiss4d(reverse=False) # Reference twiss
tw_b12 = {'b1': tw_b1, 'b2': tw_b2}

# errors back on
for nn in tt_err_knobs.name:
    env[nn] = tt_err_knobs['value', nn]

# Local correction of IR15 multipole errors
xmlhc.correct_ir_errors(env, twiss_b1=tw_b1, twiss_b2=tw_b2,
                        corrections=config['ir_corrections'])

# Spool piece correctors (MCS, MC0, MCD)
xmlhc.set_arc_spool_piece_correctors(env, twiss_b1=tw_b1, twiss_b2=tw_b2)

# k1s local + global correction (uses MQS)
xmlhc.correct_k1s(env, twiss_b1=tw_b1, twiss_b2=tw_b2)

# k2s local + global correction (uses MSS)
xmlhc.correct_k2s(env, twiss_b1=tw_b1, twiss_b2=tw_b2)

# Save line with corrections
env.to_json('lhc_multipolar_errors_corrected.json')
