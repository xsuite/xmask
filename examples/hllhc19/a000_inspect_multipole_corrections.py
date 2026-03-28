import xtrack as xt
import xmask as xm
import xobjects as xo
import numpy as np

lhc = xt.load("lhc_thick_test_04_tuned_and_leveled_bb_on.json")

# Check that errors on a few magnet types are present,
# and that the first two orders are zero (consistently with setup)
assert np.max(np.abs(lhc['mb.a12r4.b2'].knl_rel)) > 10.
assert np.all(lhc['mb.a12r4.b2'].knl_rel[:2] == 0)
assert np.max(np.abs(lhc['mb.a12r4.b2'].ksl_rel)) > 10.
assert np.all(lhc['mb.a12r4.b2'].ksl_rel[:2] == 0)

assert np.max(np.abs(lhc['mq.12r4.b2'].knl_rel)) > 10.
assert np.all(lhc['mq.12r4.b2'].knl_rel[:2] == 0)
assert np.max(np.abs(lhc['mq.12r4.b2'].ksl_rel)) > 10.
assert np.all(lhc['mq.12r4.b2'].ksl_rel[:2] == 0)

tt_triplet_quads_15 = lhc.elements.get_table().rows['mqxf.*'].rows.match(element_type='Quadrupole')
assert len(tt_triplet_quads_15) == 2 * 2 * 2 * 6 # 2 beams, 2 sides, 2 ips, 6 quads per triplet
tt_d2_15 = lhc.elements.get_table().rows['mbrd.*']
assert len(tt_d2_15) == 2 * 2 * 2 # 2 beams, 2 sides, 2 ips
for nn in list(tt_triplet_quads_15.name) + list(tt_d2_15.name):
    assert np.max(np.abs(lhc[nn].knl_rel)) > 10.
    assert np.all(lhc[nn].knl_rel[:2] == 0)
    assert np.max(np.abs(lhc[nn].ksl_rel)) > 10.
    assert np.all(lhc[nn].ksl_rel[:2] == 0)

# Check that the the triplet correctors in 1/5 are all powered
tt_vars_orig = lhc.vars.get_table()
assert np.all(np.abs(tt_vars_orig.rows['kcsx.*[l,r][1,5]'].value) > 1e-4)
assert np.all(np.abs(tt_vars_orig.rows['kcox.*[l,r][1,5]'].value) > 1e-4)
assert np.all(np.abs(tt_vars_orig.rows['kcdx.*[l,r][1,5]'].value) > 1e-4)
assert np.all(np.abs(tt_vars_orig.rows['kctx.*[l,r][1,5]'].value) > 1e-4)
assert np.all(np.abs(tt_vars_orig.rows['kcssx.*[l,r][1,5]'].value) > 1e-4)
assert np.all(np.abs(tt_vars_orig.rows['kcosx.*[l,r][1,5]'].value) > 1e-4)
assert np.all(np.abs(tt_vars_orig.rows['kctsx.*[l,r][1,5]'].value) > 1e-4)

# Check that the spool piece correctors are powered
assert np.all(np.abs(tt_vars_orig.rows['kcs.a.*'].value) > 1e-4)
assert np.all(tt_vars_orig.rows['kco.a.*'].value == 0)
assert np.all(np.abs(tt_vars_orig.rows['kcd.a.*'].value) > 1e-4)

# Check error knob behavior
lhc.set(tt_vars_orig.rows['on_error.*'], 0)
assert np.all(lhc['mb.a12r4.b2'].knl_rel == 0)
assert np.all(lhc['mb.a12r4.b2'].ksl_rel == 0)
assert np.all(lhc['mq.12r4.b2'].knl_rel == 0)
assert np.all(lhc['mq.12r4.b2'].ksl_rel == 0)
for nn in list(tt_triplet_quads_15.name) + list(tt_d2_15.name):
    assert np.all(lhc[nn].knl_rel == 0)
    assert np.all(lhc[nn].ksl_rel == 0)

# Check correction knob behavior
lhc.set(tt_vars_orig.rows['on_corr.*'], 0)
tt_vars = lhc.vars.get_table()
assert np.all(tt_vars.rows['kcsx.*[l,r][1,5]'].value == 0)
assert np.all(tt_vars.rows['kcox.*[l,r][1,5]'].value == 0)
assert np.all(tt_vars.rows['kcdx.*[l,r][1,5]'].value == 0)
assert np.all(tt_vars.rows['kctx.*[l,r][1,5]'].value == 0)
assert np.all(tt_vars.rows['kcssx.*[l,r][1,5]'].value == 0)
assert np.all(tt_vars.rows['kcosx.*[l,r][1,5]'].value == 0)
assert np.all(tt_vars.rows['kctsx.*[l,r][1,5]'].value == 0)
assert np.all(tt_vars.rows['kcs.a.*'].value == 0)
assert np.all(tt_vars.rows['kco.a.*'].value == 0)
assert np.all(tt_vars.rows['kcd.a.*'].value == 0)

# Restore knobs
lhc.vars.update(tt_vars_orig.rows['on_corr_.*|on_error_.*'].to_dict())

# Read config file
with open('config.yaml','r') as fid:
    config = xm.yaml.load(fid)

lhc['beambeam_scale'] = 0 # Beam beam off

tw1 = lhc.b1.twiss()
tw2 = lhc.b2.twiss(reverse=True)

xo.assert_allclose(tw1.qx, 62.31, atol=1e-5)
xo.assert_allclose(tw1.qy, 60.32, atol=1e-5)
xo.assert_allclose(tw2.qx, 62.31, atol=1e-5)
xo.assert_allclose(tw2.qy, 60.32, atol=1e-5)
xo.assert_allclose(tw1.dqx, 5, atol=0.05)
xo.assert_allclose(tw1.dqy, 6, atol=0.05)
xo.assert_allclose(tw2.dqx, 5, atol=0.05)
xo.assert_allclose(tw2.dqy, 6, atol=0.05)
xo.assert_allclose(tw1.c_minus, 0, atol=2e-4)
xo.assert_allclose(tw2.c_minus, 0, atol=2e-4)

xo.assert_allclose(tw1['px', 'ip1'], 250e-6, atol=5e-7)
xo.assert_allclose(tw2['px', 'ip1'], -250e-6, atol=5e-7)
xo.assert_allclose(tw1['py', 'ip1'], 0, atol=5e-7)
xo.assert_allclose(tw2['py', 'ip1'], 0, atol=5e-7)
xo.assert_allclose(tw1['x', 'ip1'], 0, atol=1e-7)
xo.assert_allclose(tw2['x', 'ip1'], 0, atol=1e-7)
xo.assert_allclose(tw1['y', 'ip1'], 0, atol=1e-7)
xo.assert_allclose(tw2['y', 'ip1'], 0, atol=1e-7)

xo.assert_allclose(tw1['py', 'ip5'], 250e-6, atol=5e-7)
xo.assert_allclose(tw2['py', 'ip5'], -250e-6, atol=5e-7)
xo.assert_allclose(tw1['px', 'ip5'], 0, atol=5e-7)
xo.assert_allclose(tw2['px', 'ip5'], 0, atol=5e-7)
xo.assert_allclose(tw1['x', 'ip5'], 0, atol=1e-7)
xo.assert_allclose(tw2['x', 'ip5'], 0, atol=1e-7)
xo.assert_allclose(tw1['y', 'ip5'], 0, atol=1e-7)
xo.assert_allclose(tw2['y', 'ip5'], 0, atol=1e-7)

xo.assert_allclose(lhc['on_x8v'], -200, rtol=1e-10)
xo.assert_allclose(lhc['on_x2v'], -170, rtol=1e-10)
with xt.line._temp_knobs(lhc, dict(on_disp=0, on_x8v=0, on_x2v=0,
                                    on_sep8h=0, on_sep8v=0,
                                    on_sep2h=0, on_sep2v=0)):
    xo.assert_allclose(lhc['on_x8v'], 0, rtol=1e-10)
    xo.assert_allclose(lhc['on_x2v'], 0, rtol=1e-10)
    tw1_internal_cross_28 = lhc.b1.twiss()
    tw2_internal_cross_28 = lhc.b2.twiss(reverse=True)
xo.assert_allclose(lhc['on_x8v'], -200, rtol=1e-10)
xo.assert_allclose(lhc['on_x2v'], -170, rtol=1e-10)

xo.assert_allclose(tw1_internal_cross_28['px', 'ip8'], 134e-6, atol=2e-6)
xo.assert_allclose(tw2_internal_cross_28['px', 'ip8'], -134e-6, atol=2e-6)
xo.assert_allclose(tw1['px', 'ip8'], 134e-6, atol=2e-6)
xo.assert_allclose(tw2['px', 'ip8'], -134e-6, atol=2e-6)
xo.assert_allclose(tw1_internal_cross_28['py', 'ip8'], 0, atol=3e-6)
xo.assert_allclose(tw2_internal_cross_28['py', 'ip8'], 0, atol=3e-6)

xo.assert_allclose(tw1_internal_cross_28['py', 'ip2'], 70e-6, atol=2e-6)
xo.assert_allclose(tw2_internal_cross_28['py', 'ip2'], -70e-6, atol=2e-6)
xo.assert_allclose(tw1['py', 'ip2'], -170e-6 + 70e-6, atol=2e-6)
xo.assert_allclose(tw2['py', 'ip2'], 170e-6 -70e-6, atol=2e-6)
xo.assert_allclose(tw1_internal_cross_28['px', 'ip2'], 0, atol=2e-6)
xo.assert_allclose(tw2_internal_cross_28['px', 'ip2'], 0, atol=2e-6)

# Check that the bumps from the experimental dipoles are closed (no angle in at the triplets)
xo.assert_allclose(tw1_internal_cross_28.rows[
    ['bpms.2l8.b1', 'bpmsw.1l8.b1', 'bpmsw.1r8.b1', 'bpms.2r8.b1']].px,
    0, atol=3e-6)
xo.assert_allclose(tw2_internal_cross_28.rows[
    ['bpms.2l8.b2', 'bpmsw.1l8.b2', 'bpmsw.1r8.b2', 'bpms.2r8.b2']].px,
    0, atol=3e-6)
xo.assert_allclose(tw1_internal_cross_28.rows[
    ['bpms.2l2.b1', 'bpmsw.1l2.b1', 'bpmsw.1r2.b1', 'bpms.2r2.b1']].px,
    0, atol=3e-6)
xo.assert_allclose(tw2_internal_cross_28.rows[
    ['bpms.2l2.b2', 'bpmsw.1l2.b2', 'bpmsw.1r2.b2', 'bpms.2r2.b2']].px,
    0, atol=3e-6)

prrrr


# Remove corrections that are not valid with flat orbit
lhc['on_corr_co'] = 0
lhc['cmis.b1_op'] = 0
lhc['cmis.b2_op'] = 0
lhc['cmrs.b1_op'] = 0
lhc['cmrs.b2_op'] = 0

# Flat orbit
vars_to_zero = config['knobs_to_zero_for_flat_orbit']
tt_to_zero = lhc.vars.get_table(expr_obj=True).rows[vars_to_zero]
lhc.set(tt_to_zero, 0)

env_test = lhc

global_chrom_coupling = {}
global_chrom_coupling_no_corr = {}
local_chrom_coupling = {}
for line_name in ['b1', 'b2']:

    line_test = env_test[line_name]
    # Chromatic coupling integrand
    tw_test = line_test.twiss4d(strengths=True)

    tw_test['chrom_coupl_source'] = (tw_test.k2sl * tw_test.dx * np.sqrt(tw_test.betx * tw_test.bety)
                                * np.exp(1j*2*np.pi*(tw_test.mux - tw_test.muy)))

    # Chromatic coupling integral arc by arc
    chrom_coupling_integ_test = {}
    chrom_coupling_integ_ref = {}
    for arc in ['12', '23', '34', '45', '56', '67', '78', '81']:
        if line_name == 'b1':
            start = f's.ds.r{arc[0]}.b1'
            end = f'e.ds.l{arc[1]}.b1'
        else:
            start = f'e.ds.l{arc[1]}.b2'
            end = f's.ds.r{arc[0]}.b2'
        tt_test_arc = tw_test.rows[start:end]

        chrom_coupling_integ_test[arc] = np.sum(tt_test_arc.chrom_coupl_source)

    twom_test = line_test.twiss4d(coupling_edw_teng=True, delta0=0.5e-3)

    nlchr_test = line_test.get_non_linear_chromaticity(delta0_range=(-5e-4, 5e-4), num_delta=20)

    cminus_delta_test = np.array([ttt.c_minus for ttt in nlchr_test.twiss])

    # Check on local chromatic coupling integrals
    arc_names = list(chrom_coupling_integ_test.keys())
    arc_chrom_coupl_test = [chrom_coupling_integ_test[arc_name] for arc_name in arc_names]

    # Measure chromatic coupling without correction
    tt_on_corr = env_test.vars.get_table().rows['on_corr_k2s.*']
    env_test.set(tt_on_corr.name, 0)
    nlchr_test_no_corr = line_test.get_non_linear_chromaticity(num_delta=20, delta0_range=(-5e-4, 5e-4))
    env_test.set(tt_on_corr.name, 1)
    cminus_delta_test_no_corr = np.array([ttt.c_minus for ttt in nlchr_test_no_corr.twiss])

    global_chrom_coupling[line_name] = cminus_delta_test
    local_chrom_coupling[line_name] = arc_chrom_coupl_test
    global_chrom_coupling_no_corr[line_name] = cminus_delta_test_no_corr

assert np.max(global_chrom_coupling_no_corr['b1']) > 2e-3
assert np.max(global_chrom_coupling_no_corr['b2']) > 2e-3
assert np.all(np.abs(global_chrom_coupling['b1']) < 3e-4)
assert np.all(np.abs(global_chrom_coupling['b2']) < 3e-4)

# Check effect of mcs on chromaticity
lhc.set(lhc.vars.get_table().rows['on_corr_kcs.*'], 0)
tw_b1_mcs_off = lhc.b1.twiss4d()
tw_b2_mcs_off = lhc.b2.twiss4d()
lhc.set(lhc.vars.get_table().rows['on_corr_kcs.*'], 1)
tw_b1_mcs_on = lhc.b1.twiss4d()
tw_b2_mcs_on = lhc.b2.twiss4d()

assert np.abs(tw_b1_mcs_off.dqx) > 100
assert np.abs(tw_b1_mcs_off.dqy) > 100
assert np.abs(tw_b1_mcs_on.dqx) < 20
assert np.abs(tw_b1_mcs_on.dqy) < 20

# Back to clean machine
lhc.set(tt_vars.rows['on_.*'], 0)
lhc.set(tt_vars.rows['cmis.*|cmrs.*'], 0)
lhc.set(tt_vars.rows['dqx.*|dqy.*'], 0)
lhc.set(tt_vars.rows['dqp.*'], 0)

tw1 = lhc.b1.twiss(strengths=True)
tw2 = lhc.b2.twiss(strengths=True)

for tw in [tw1, tw2]:
    xo.assert_allclose(tw.x, 0, atol=1e-10)
    xo.assert_allclose(tw.px, 0, atol=1e-10)
    xo.assert_allclose(tw.y, 0, atol=1e-10)
    xo.assert_allclose(tw.py, 0, atol=1e-10)
    xo.assert_allclose(tw.qx, 62.31, atol=1e-6)
    xo.assert_allclose(tw.qy, 60.32, atol=1e-6)
    xo.assert_allclose(tw.dqx, 0, atol=1e-3)
    xo.assert_allclose(tw.dqy, 0, atol=1e-3)
    xo.assert_allclose(tw.c_minus, 0, atol=1e-4)
    xo.assert_allclose(tw.qs, 0.0021243, atol=1e-5)
    xo.assert_allclose(tw.rows[['ip1', 'ip2', 'ip5', 'ip8']].betx,
                       [0.15, 10, 0.15, 1.5], rtol=1e-4)
    xo.assert_allclose(tw.k1sl, 0, atol=1e-12)
    xo.assert_allclose(tw.k2sl, 0, atol=1e-12)
    xo.assert_allclose(tw.k3sl, 0, atol=1e-12)
    xo.assert_allclose(tw.k4l, 0, atol=1e-12)
    xo.assert_allclose(tw.k4sl, 0, atol=1e-12)
    xo.assert_allclose(tw.k5l, 0, atol=1e-12)
    xo.assert_allclose(tw.k5sl, 0, atol=1e-12)


import matplotlib.pyplot as plt
plt.close('all')

for ii, line_name in enumerate(['b1', 'b2']):

    c_minus_delta_test = global_chrom_coupling[line_name]
    c_minus_delta_test_no_corr = global_chrom_coupling_no_corr[line_name]
    arc_chrom_coupl_test = local_chrom_coupling[line_name]

    plt.figure(1 + 10 * ii)
    plt.plot(1e3 * nlchr_test_no_corr.delta0, c_minus_delta_test_no_corr, label='Test no corr', linestyle='dashed')
    plt.plot(1e3 * nlchr_test.delta0, c_minus_delta_test, label='Test')
    plt.xlabel(r'$\delta_0$ [10$^{-3}$]')
    plt.ylabel('C-')
    plt.legend()
    plt.suptitle(line_name)

    # Bar plot of chromatic coupling integral arc by arc
    plt.figure(2 + 10 * ii)

    x = np.arange(len(arc_names))
    plt.bar(x - 0.2, np.abs(arc_chrom_coupl_test), width=0.4, label='Test')
    plt.xticks(x, arc_names)
    plt.xlabel('Arc name')
    plt.ylabel('Abs of chromatic coupling integral')
    plt.legend()
    plt.suptitle(line_name)

plt.show()