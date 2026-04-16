import xtrack as xt
import xmask as xm
import xmask.lhc as xmlhc

# Read config file
with open('config.yaml','r') as fid:
    config = xm.yaml.load(fid)

lhc = xt.load(f'lhc_{config["label"]}_00_prepared.json')

#############################
# Install multipolar errors #
#############################

apply_multipolar_errors_config = config['apply_multipolar_errors']

if apply_multipolar_errors_config:
    # Read the configuration from the yaml
    err_conf = apply_multipolar_errors_config.pop('_config_')
    min_order = err_conf['min_order']
    max_order = err_conf['max_order']

    # Apply the errors
    for knob_name, json_file in apply_multipolar_errors_config.items():
        print(f'Applying multipolar errors from file to create knob {knob_name}')
        # Read the file
        multipole_errors = xt.json.load(json_file)
        for line_name in ['b1', 'b2']:
            line = lhc[line_name]
            # Apply the errors in the line
            xm.set_multipole_errors_in_line(line, multipole_errors,
                                    min_order=min_order, max_order=max_order,
                                    error_knob_name=knob_name,
                                    append_order_to_knob_name=True)


#####################################
# Corrections for multipolar errors #
#####################################


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
                        corrections=config['ir_corrections'])

# Spool piece correctors (MCS, MC0, MCD)
xmlhc.set_arc_spool_piece_correctors(lhc, twiss_b1=tw_b1, twiss_b2=tw_b2,
                                        use_mcs=True, use_mcd=True,
                                        use_mco=False) # dead circuits

# k1s local + global correction (uses MQS)
xmlhc.correct_k1s(lhc, twiss_b1=tw_b1, twiss_b2=tw_b2, feed_down=False)

# k2s local + global correction (uses MSS)
xmlhc.correct_k2s(lhc, twiss_b1=tw_b1, twiss_b2=tw_b2, feed_down=False)

# Back to orbit with bumps
for nn in tt_to_zero.name:
    expr_obj = tt_to_zero['expr_obj', nn]
    val = tt_to_zero['value', nn]
    if expr_obj is not None:
        lhc[nn] = expr_obj
    else:
        lhc[nn] = val

lhc.to_json(f'lhc_{config["label"]}_01_multipolar_errors_corrected.json')
