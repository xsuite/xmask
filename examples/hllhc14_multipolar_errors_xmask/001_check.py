import xtrack as xt
import xobjects as xo
import numpy as np

env_test = xt.load('lhc_arc_errors.json')
env_ref = xt.load('../hllhc14_multipolar_errors_legacy/'
                   'collider_errors_on_corrections_off.json')

# To be in the same conditions as the reference
env_test['on_error_arc_k0'] = 0
env_test['on_error_arc_k0s'] = 0
env_test['on_error_arc_k1'] = 0
env_test['on_error_arc_k1s'] = 0

for line_to_check in ['lhcb1', 'lhcb2']:
    line_test = env_test[line_to_check]
    line_ref = env_ref[line_to_check]

    tt_test = line_test.get_table(attr=True)
    tt_ref = line_ref.get_table(attr=True)

    tt_test_quad = tt_test.rows['mq.*']
    tt_ref_quad = tt_ref.rows['mq.*']

    max_order = 18


    for arc in ['12', '23', '34', '45', '56', '67', '78', '81']:
        if line_to_check == 'lhcb1':
            start = f's.ds.r{arc[0]}.b1'
            end = f'e.ds.l{arc[1]}.b1'
        else:
            start = f'e.ds.l{arc[1]}.b2'
            end = f's.ds.r{arc[0]}.b2'
        tt_test_arc = tt_test.rows[start:end]
        tt_ref_arc = tt_ref.rows[start:end]

        assert len(tt_test_arc) > 100
        assert len(tt_ref_arc) > 100

        for nn in tt_test_arc.name:
            print(f'Checking {nn}               ', end='\r', flush=True)

            if nn in {'ms.29l2.b2', 'ms.33l2.b2', 'ms.33r1.b2', 'ms.29r1.b2',
                      'ms.28l3.b2', 'ms.32l3.b2', 'ms.34l3.b2', 'ms.30r2.b2',
                      'ms.29l4.b2', 'ms.33l4.b2', 'ms.33r3.b2', 'ms.29r3.b2',
                      'ms.28l5.b2', 'ms.32l5.b2', 'ms.34l5.b2', 'ms.30r4.b2',
                      'ms.29l6.b2', 'ms.33l6.b2', 'ms.33r5.b2'}:
                # issue in the legacy macros, errors non applied to this magnet
                assert np.all(line_ref[nn].get_total_knl_ksl()[0][3:] == 0)
                assert np.all(line_ref[nn].get_total_knl_ksl()[1] == 0)
                continue
            if nn in {'mco.b30r3.b2'}:
                "magnet seems to be off"
                assert np.all(line_ref[nn].knl == 0)
                assert np.all(line_ref[nn].ksl == 0)
                assert np.all(line_test[nn].knl == 0)
                assert np.all(line_test[nn].ksl == 0)
                continue


            if hasattr(line_ref[nn], 'knl'):
                for ii in range(len(line_ref[nn].knl)):
                    knl_tot_nn, ksl_tot_nn = line_test[nn].get_total_knl_ksl()
                    xo.assert_allclose(knl_tot_nn[ii], line_ref[nn].knl[ii],
                                    rtol=1e-10, atol=1e-10)
                    xo.assert_allclose(ksl_tot_nn[ii], line_ref[nn].ksl[ii],
                                        rtol=1e-10, atol=1e-10)
