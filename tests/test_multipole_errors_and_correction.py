import xtrack as xt
import xmask.lhc as xmlhc

from pathlib import Path

test_data_dir = Path(__file__).parent.parent / "test_data"

fname_rotations = (test_data_dir /
    "hllhc19/prepare_multipolar_errors/multipole_errors_pre_hllhc/magnet_orientation.tab")
fname_err_table = (test_data_dir /
    "hllhc19/prepare_multipolar_errors/multipole_errors_pre_hllhc/collision_errors-emfqcs-6.tfs")

min_order = 0
max_order = 15

multipole_errors_arcs, tt_err_arc = xmlhc.load_wise_table_arc_magnets(
    fname_err_table=fname_err_table,
    fname_rotations=fname_rotations,
    min_order=min_order, max_order=max_order)
