import xtrack as xt
import xmask as xm


# Load multipolar errors from json files
multipole_errors_arc = xt.json.load(
    'multipole_errors_pre_hllhc/multipole_errors_arc.json')
multipole_errors_triplet_ir15 = xt.json.load(
    'multipole_errors_hllhc_ir15/multipole_errors_inner_triplet_d1_ir15.json')
multipole_errors_d2_ir15 = xt.json.load(
    'multipole_errors_hllhc_ir15/multipole_errors_d2_ir15.json')


# Association knob_name -> multipole errors
multipole_errors_to_apply = {
    'on_error_arc': multipole_errors_arc,
    'on_error_triplets_ir15': multipole_errors_triplet_ir15,
    'on_error_d2_ir15': multipole_errors_d2_ir15,
}


# Get a collider model
env = xt.load(
    '../hllhc14_multipolar_errors_legacy/collider_errors_off_corrections_off.json')


# Apply error in lines

min_order = 0
max_order = 15

for knob_name, multipole_errors in multipole_errors_to_apply.items():
    for line_name in ['lhcb1', 'lhcb2']:
        line = env[line_name]
        xm.set_multipole_errors_in_line(line, multipole_errors,
                                min_order=min_order, max_order=max_order,
                                error_knob_name=knob_name,
                                append_order_to_knob_name=True)

env.to_json('lhc_multipolar_errors.json')
