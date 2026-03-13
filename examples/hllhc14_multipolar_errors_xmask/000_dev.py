import xtrack as xt
from load_wise import load_wise_table_arc_magnets, set_multipole_errors_in_line

env = xt.load('../hllhc14_multipolar_errors_legacy/collider_errors_off_corrections_off.json')


fname_rotations = './magnet_orientation.tab'
# fname_err_table = '../../xmask/lhc/lhcerrors/LHC/fidel/collision_errors-emfqcs-6.tfs'
fname_err_table = 'collision_errors-emfqcs-6.tfs'

min_order = 0
max_order = 15

multipole_errors, tt_err_arc = load_wise_table_arc_magnets(
    fname_err_table=fname_err_table,
    fname_rotations=fname_rotations,
    min_order=min_order, max_order=max_order)

for line_name in ['lhcb1', 'lhcb2']:
    line = env[line_name]

    set_multipole_errors_in_line(line, multipole_errors,
                                min_order=min_order, max_order=max_order,
                                error_knob_name='on_error_arc',
                                append_order_to_knob_name=True)

env.to_json('lhc_arc_errors.json')
