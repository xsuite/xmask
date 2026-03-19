import xtrack as xt
import xobjects as xo
import numpy as np
import xmask.lhc as xmlhc

env_test = xt.load('lhc_multipolar_errors.json')

line_b1_ref = xt.load(
    '../../../20260318_andrea_fornara_errors/'
    'sorting/triplet_errors/sorting/full_vertical_data_sorting/results/'
    'line_lhcb1_perm6.json'
)
line_b2_ref = xt.load(
    '../../../20260318_andrea_fornara_errors/'
    'sorting/triplet_errors/sorting/full_vertical_data_sorting/results/'
    'line_lhcb2_perm6.json'
)
env_ref = xt.Environment(lines={'lhcb1': line_b1_ref, 'lhcb2': line_b2_ref})

# Check  b1/b2 consistency in env_test
tt_b1_mqxf = env_test['lhcb1'].get_table(attr=True).rows['mqxf.*']
for nn in tt_b1_mqxf.name:
    if '..fl' in nn or '..fr' in nn:
        continue # skip edge lenses
    if isinstance(env_test[nn], xt.Marker):
        continue # skip markers
    ele_b1 = env_test['lhcb1'][nn]
    ele_b2 = env_test['lhcb2'][nn.replace('/lhcb1', '/lhcb2')]
    xmlhc.assert_are_same_multipoles_b1_b2(ele_b1, ele_b2, rtol=1e-12, atol=1e-12)


for line_to_check in ['lhcb1']: #, 'lhcb2']: # for b2 there is a bug in the reference

    line_test = env_test[line_to_check]
    line_ref = env_ref[line_to_check]

    line_test.cycle('ip7')
    line_ref.cycle('ip7')

    # Check b1/b2 consistency
    line_test = env_test['lhcb2']

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