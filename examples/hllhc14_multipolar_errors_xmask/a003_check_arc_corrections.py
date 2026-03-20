import xtrack as xt
import matplotlib.pyplot as plt
import numpy as np
import xobjects as xo

env_test = xt.load('lhc_multipolar_errors_corrected.json')
env_ref = xt.load('../hllhc14_multipolar_errors_legacy/'
                   'collider_errors_on_corrections_on.json')

# Check spool pieces
tt_kcs_test = env_test.vars.get_table().rows.match(r'kcs\..*').rows.match_not('.*_from_.*')
tt_kcs_ref = env_ref.vars.get_table().rows.match(r'kcs\..*').rows.match_not('.*_from_.*')
assert np.all(tt_kcs_test.name == tt_kcs_ref.name)
xo.assert_allclose(tt_kcs_test.value, tt_kcs_ref.value, rtol=0.02)

tt_kco_test = env_test.vars.get_table().rows.match(r'kco\..*').rows.match_not('.*_from_.*')
tt_kco_ref = env_ref.vars.get_table().rows.match(r'kco\..*').rows.match_not('.*_from_.*')
assert np.all(tt_kco_test.name == tt_kco_ref.name)
xo.assert_allclose(tt_kco_test.value, tt_kco_ref.value, rtol=0.001)

tt_kcd_test = env_test.vars.get_table().rows.match(r'kcd\..*').rows.match_not('.*_from_.*')
tt_kcd_ref = env_ref.vars.get_table().rows.match(r'kcd\..*').rows.match_not('.*_from_.*')
assert np.all(tt_kcd_test.name == tt_kcd_ref.name)
xo.assert_allclose(tt_kcd_test.value, tt_kcd_ref.value, rtol=0.001,
                   atol=0.03*np.max(np.abs(tt_kcd_ref.value)))

global_chrom_coupling = {}
global_chrom_coupling_no_corr = {}
global_chrom_coupling_ref = {}
local_chrom_coupling = {}
local_chrom_coupling_ref = {}
for line_name in ['lhcb1', 'lhcb2']:

    line_test = env_test[line_name]
    line_ref = env_ref[line_name]

    tt_ref = line_ref.get_table(attr=True)
    tt_test = line_test.get_table(attr=True)

    # patch hxl
    for nn in tt_ref.name:
        if nn == '_end_point':
            continue
        if hasattr(line_ref[nn], 'hxl'):
            line_ref[nn].hxl = line_ref[nn].knl[0]
            line_ref[nn].shift_x = 0 # Shifts are not applied in the test line

    # kill IR non-linearities in reference machine (for comparison)
    for ipn in [1, 2, 5, 8]:
        if line_name == 'lhcb1':
            start_ir = f'e.ds.l{ipn}.b1'
            end_ir = f's.ds.r{ipn}.b1'
        else:
            start_ir = f's.ds.r{ipn}.b2'
            end_ir = f'e.ds.l{ipn}.b2'
        tt = line_ref.get_table()
        tt_ir = tt.rows[start_ir:end_ir]
        for nn in tt_ir.name:
            if hasattr(line_ref[nn], 'knl'):
                line_ref[nn].knl[2:] = 0
            if hasattr(line_ref[nn], 'ksl'):
                line_ref[nn].ksl[1:] = 0

    # Chromatic coupling integrand
    tw_test = line_test.twiss4d(strengths=True)
    tw_ref = line_ref.twiss4d(strengths=True)

    for ttww in [tw_test, tw_ref]:
        ttww['chrom_coupl_source'] = (ttww.k2sl * ttww.dx * np.sqrt(ttww.betx * ttww.bety)
                                    * np.exp(1j*2*np.pi*(ttww.mux - ttww.muy)))

    # Chromatic coupling integral arc by arc
    chrom_coupling_integ_test = {}
    chrom_coupling_integ_ref = {}
    for arc in ['12', '23', '34', '45', '56', '67', '78', '81']:
        if line_name == 'lhcb1':
            start = f's.ds.r{arc[0]}.b1'
            end = f'e.ds.l{arc[1]}.b1'
        else:
            start = f'e.ds.l{arc[1]}.b2'
            end = f's.ds.r{arc[0]}.b2'
        tt_test_arc = tw_test.rows[start:end]
        tt_ref_arc = tw_ref.rows[start:end]

        chrom_coupling_integ_test[arc] = np.sum(tt_test_arc.chrom_coupl_source)
        chrom_coupling_integ_ref[arc] = np.sum(tt_ref_arc.chrom_coupl_source)

    twom_test = line_test.twiss4d(coupling_edw_teng=True, delta0=0.5e-3)
    twom_ref = line_ref.twiss4d(coupling_edw_teng=True, delta0=0.5e-3)

    nlchr_test = line_test.get_non_linear_chromaticity(delta0_range=(-5e-4, 5e-4), num_delta=20)
    nlchr_ref = line_ref.get_non_linear_chromaticity(delta0_range=(-5e-4, 5e-4), num_delta=20)

    cminus_delta_test = np.array([ttt.c_minus for ttt in nlchr_test.twiss])
    cminus_delta_ref = np.array([ttt.c_minus for ttt in nlchr_ref.twiss])

    # Check on global chromatic coupling
    assert np.all(cminus_delta_test < cminus_delta_ref)

    # Check on local chromatic coupling integrals
    arc_names = list(chrom_coupling_integ_test.keys())
    arc_chrom_coupl_test = [chrom_coupling_integ_test[arc_name] for arc_name in arc_names]
    arc_chrom_coupl_ref = [chrom_coupling_integ_ref[arc_name] for arc_name in arc_names]
    assert np.abs(np.mean(arc_chrom_coupl_test)) < np.abs(np.mean(arc_chrom_coupl_ref))
    assert np.all(np.abs(arc_chrom_coupl_test) < 1.05 * np.max(np.abs(arc_chrom_coupl_ref)))

    # Measure chromatic coupling without correction
    tt_on_corr = env_test.vars.get_table().rows['on_corr_k2sl.*']
    assert len(tt_on_corr) == 9
    env_test.set(tt_on_corr.name, 0)
    nlchr_test_no_corr = line_test.get_non_linear_chromaticity(num_delta=20, delta0_range=(-5e-4, 5e-4))
    env_test.set(tt_on_corr.name, 1)
    cminus_delta_test_no_corr = np.array([ttt.c_minus for ttt in nlchr_test_no_corr.twiss])

    global_chrom_coupling[line_name] = cminus_delta_test
    local_chrom_coupling[line_name] = arc_chrom_coupl_test
    global_chrom_coupling_no_corr[line_name] = cminus_delta_test_no_corr
    global_chrom_coupling_ref[line_name] = cminus_delta_ref
    local_chrom_coupling_ref[line_name] = arc_chrom_coupl_ref


plt.close('all')

for ii, line_name in enumerate(['lhcb1', 'lhcb2']):

    c_minus_delta_test = global_chrom_coupling[line_name]
    c_minus_delta_test_no_corr = global_chrom_coupling_no_corr[line_name]
    c_minus_delta_ref = global_chrom_coupling_ref[line_name]
    arc_chrom_coupl_test = local_chrom_coupling[line_name]
    arc_chrom_coupl_ref = local_chrom_coupling_ref[line_name]

    plt.figure(1 + 10 * ii)
    plt.plot(1e3 * nlchr_test_no_corr.delta0, c_minus_delta_test_no_corr, label='Test no corr', linestyle='dashed')
    plt.plot(1e3 * nlchr_test.delta0, c_minus_delta_test, label='Test')
    plt.plot(1e3 * nlchr_ref.delta0, c_minus_delta_ref, label='Reference', linestyle='dashed')
    plt.xlabel(r'$\delta_0$ [10$^{-3}$]')
    plt.ylabel('C-')
    plt.legend()
    plt.suptitle(line_name)

    # Bar plot of chromatic coupling integral arc by arc
    plt.figure(2 + 10 * ii)

    x = np.arange(len(arc_names))
    plt.bar(x - 0.2, np.abs(arc_chrom_coupl_test), width=0.4, label='Test')
    plt.bar(x + 0.2, np.abs(arc_chrom_coupl_ref), width=0.4, label='Reference')
    plt.xticks(x, arc_names)
    plt.xlabel('Arc name')
    plt.ylabel('Abs of chromatic coupling integral')
    plt.legend()
    plt.suptitle(line_name)

plt.show()