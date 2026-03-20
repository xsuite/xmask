import xtrack as xt
import xmask as xm

# Read config file
with open('config.yaml','r') as fid:
    config = xm.yaml.load(fid)

lhc = xt.load('collider_00_prepared.json')

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

# Force the knobs settings (on_error_... might be forced to 1 by the error,
# installation and we want the user setting, it present to be applied on top of that)
for knob_name, knob_value in config['knob_settings'].items():
    lhc[knob_name] = knob_value

lhc.to_json('collider_01_multipolar_errors.json')
