import xtrack as xt
import xmask.lhc as xmlhc

env = xt.load('lhc_multipolar_errors.json')

# Status of error knobs
tt_err_knobs = env.vars.get_table().rows[r'on_error_arc.*']
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

# Spool piece correctors
xmlhc.set_arc_spool_piece_correctors(env, twiss_b1=tw_b1, twiss_b2=tw_b2)