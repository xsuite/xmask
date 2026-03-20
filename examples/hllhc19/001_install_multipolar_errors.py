import xtrack as xt
import xmask as xm

# Read config file
with open('config.yaml','r') as fid:
    config = xm.yaml.load(fid)

env = xt.load('collider_00_prepared.json')

apply_multipolar_errors_config = config['apply_multipolar_errors']

if apply_multipolar_errors_config:
    config = apply_multipolar_errors_config.pop('_config_')
    min_order = config['min_order']
    max_order = config['max_order']
    for knob_name, json_file in apply_multipolar_errors_config.items():
        print(f'Applying multipolar errors from file to create knob {knob_name}')
        multipole_errors = xt.json.load(json_file)
        for line_name in ['b1', 'b2']:
            line = env[line_name]
            xm.set_multipole_errors_in_line(line, multipole_errors,
                                    min_order=min_order, max_order=max_order,
                                    error_knob_name=knob_name,
                                    append_order_to_knob_name=True)


# # Switch off errors of order 0 and 1
# env['on_error_arc_k0'] = 0
# env['on_error_arc_k0s'] = 0
# env['on_error_arc_k1'] = 0
# env['on_error_arc_k1s'] = 0
# env['on_error_triplets_ir15_k0'] = 0
# env['on_error_triplets_ir15_k0s'] = 0
# env['on_error_triplets_ir15_k1'] = 0
# env['on_error_triplets_ir15_k1s'] = 0
# env['on_error_d2_ir15_k0'] = 0
# env['on_error_d2_ir15_k0s'] = 0
# env['on_error_d2_ir15_k1'] = 0
# env['on_error_d2_ir15_k1s'] = 0

# env.to_json('lhc_multipolar_errors.json')
