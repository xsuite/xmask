import xtrack as xt
import xmask as xm
import xmask.lhc as xmlhc

# Read config file
with open('config.yaml','r') as fid:
    config = xm.yaml.load(fid)

# Load the line from previous step
lhc = xt.load('collider_01_multipolar_errors.json')

# Force the knobs settings (on_error_... might be forced to 1 by the error,
# installation and we want the user setting, it present to be applied on top of that)
for knob_name, knob_value in config['knob_settings'].items():
    lhc[knob_name] = knob_value

# Go to flat orbit
vars_to_zero = config['knobs_to_zero_for_flat_orbit']
tt_to_zero = lhc.vars.get_table(expr_obj=True).rows[vars_to_zero]
lhc.set(tt_to_zero, 0)

# Status of error knobs
tt_err_knobs = lhc.vars.get_table().rows[r'on_error_.*']
print("Error knobs in the environment:")
tt_err_knobs.show()

# Errors off to get reference twiss
lhc.set(tt_err_knobs.name, 0)
tw_b1 = lhc['b1'].twiss4d(reverse=False)
tw_b2 = lhc['b2'].twiss4d(reverse=False)
tw_b12 = {'b1': tw_b1, 'b2': tw_b2}

# errors back on
for nn in tt_err_knobs.name:
    lhc[nn] = tt_err_knobs['value', nn]

# Local correction of IR15 multipole errors
xmlhc.correct_ir_errors(lhc, twiss_b1=tw_b1, twiss_b2=tw_b2,
                        corrections=xmlhc.DEFAULT_IR15_CORRECTIONS)

# Spool piece correctors (MCS, MC0, MCD)
xmlhc.set_arc_spool_piece_correctors(lhc, twiss_b1=tw_b1, twiss_b2=tw_b2)

# k1s local + global correction (uses MQS)
xmlhc.correct_k1s(lhc, twiss_b1=tw_b1, twiss_b2=tw_b2)

# k2s local + global correction (uses MSS)
xmlhc.correct_k2s(lhc, twiss_b1=tw_b1, twiss_b2=tw_b2)