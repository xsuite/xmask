import xtrack as xt
import xobjects as xo
import numpy as np

# env_test = xt.load('lhc_arc_errors_arc_and_triplets15.json')
# line_test = env_test['lhcb1']
# line_ref = xt.load(
#     '../../../20260318_andrea_fornara_errors/'
#     'sorting/triplet_errors/sorting/full_vertical_data_sorting/results/'
#     'line_lhcb1_perm6.json'
# )

env_test = xt.load('lhc_arc_errors_arc_and_triplets15.json')
line_test = env_test['lhcb2']
line_ref = xt.load(
    '../../../20260318_andrea_fornara_errors/'
    'sorting/triplet_errors/sorting/full_vertical_data_sorting/results/'
    'line_lhcb2_perm6.json'
)


line_test.cycle('ip7')
line_ref.cycle('ip7')

tt_test = line_test.get_table(attr=True)
tt_ref = line_ref.get_table(attr=True)

max_order = 18

tt_ref_mqxf = tt_ref.rows['mqxf.*']


for nn in tt_ref_mqxf.name:
    if '..fl' in nn or '..fr' in nn:
        continue # skip edge lenses
    print(f'Checking {nn}               ', end='\r', flush=True)

    if hasattr(line_ref[nn], 'knl'):
        knl_tot_nn, ksl_tot_nn = line_test[nn].get_total_knl_ksl()
        for ii in range(len(line_ref[nn].knl)):
            xo.assert_allclose(knl_tot_nn[ii], line_ref[nn].knl[ii],
                            rtol=0.01, atol=1e-10)
            xo.assert_allclose(ksl_tot_nn[ii], line_ref[nn].ksl[ii],
                                rtol=0.01, atol=1e-10)