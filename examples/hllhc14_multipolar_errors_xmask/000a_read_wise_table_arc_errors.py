"""
This code reads a multipolar error table in the legacy WISE format and
saves a json file with the relative multipolar strengths compatible with
xtrack multupole definition.
"""

from load_wise import load_wise_table_arc_magnets, set_multipole_errors_in_line
import xtrack as xt

fname_rotations = './magnet_orientation.tab'
fname_err_table = '../../xmask/lhc/lhcerrors/LHC/fidel/collision_errors-emfqcs-6.tfs'
# fname_err_table = 'collision_errors-emfqcs-6.tfs'

min_order = 0
max_order = 15

multipole_errors, tt_err_arc = load_wise_table_arc_magnets(
    fname_err_table=fname_err_table,
    fname_rotations=fname_rotations,
    min_order=min_order, max_order=max_order)

xt.json.dump(multipole_errors, 'multipole_errors_arc.json')