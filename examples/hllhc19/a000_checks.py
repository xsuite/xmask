import xtrack as xt
import xmask as xm
import xobjects as xo
import numpy as np

label = 'thick'

if label == 'thin':
    lhc = xt.load("lhc_thin_test_04_tuned_and_leveled_bb_on.json")
elif label == 'thick':
    lhc = xt.load("lhc_thick_test_04_tuned_and_leveled_bb_on.json")
elif label == 'legacy_thick':
    lhc = xt.load("lhc_legacy_thick_test_04_tuned_and_leveled_bb_on.json")
else:
    raise ValueError

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

tt_triplet_quads_15 = lhc.elements.get_table().rows['mqxf.*/b.*'].rows.match(element_type='Quadrupole')
assert len(tt_triplet_quads_15) == 2 * 2 * 2 * 6 # 2 beams, 2 sides, 2 ips, 6 quads per triplet
tt_d2_15 = lhc.elements.get_table().rows['mbrd.*4.*'].rows.match(element_type='RBend')
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

# Check global quantities
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

# Check orbit at experimental IPs
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

# Check orbit at other ips
for ip in ['ip3', 'ip4', 'ip6', 'ip7']:
    xo.assert_allclose(tw1['px', ip], 0, atol=5e-6)
    xo.assert_allclose(tw2['px', ip], 0, atol=5e-6)
    xo.assert_allclose(tw1['py', ip], 0, atol=5e-6)
    xo.assert_allclose(tw2['py', ip], 0, atol=5e-6)
    xo.assert_allclose(tw1['x', ip], 0, atol=5e-6)
    xo.assert_allclose(tw2['x', ip], 0, atol=5e-6)
    xo.assert_allclose(tw1['y', ip], 0, atol=5e-6)
    xo.assert_allclose(tw2['y', ip], 0, atol=5e-6)

# Check dispersions
for ip in ['ip1', 'ip2', 'ip5', 'ip8']:
    xo.assert_allclose(tw1['dx', ip], 0, atol=5e-2)
    xo.assert_allclose(tw1['dpx', ip], 0, atol=5e-2)
    xo.assert_allclose(tw1['dy', ip], 0, atol=5e-2)
    xo.assert_allclose(tw1['dpy', ip], 0, atol=5e-2)
    xo.assert_allclose(tw2['dx', ip], 0, atol=5e-2)
    xo.assert_allclose(tw2['dpx', ip], 0, atol=5e-2)
    xo.assert_allclose(tw2['dy', ip], 0, atol=5e-2)
    xo.assert_allclose(tw2['dpy', ip], 0, atol=5e-2)

# Check that dispersion degrades if disp correction is turned off
with xt.line._temp_knobs(lhc, dict(on_disp=0)):
    tw1_no_disp = lhc.b1.twiss()
    tw2_no_disp = lhc.b2.twiss(reverse=True)

assert np.abs(tw1_no_disp['dx', 'ip1']) > np.abs(tw1['dx', 'ip1']) * 4
assert np.abs(tw2_no_disp['dx', 'ip1']) > np.abs(tw2['dx', 'ip1']) * 4
assert np.abs(tw1_no_disp['dx', 'ip5']) > np.abs(tw1['dx', 'ip5']) * 4
assert np.abs(tw2_no_disp['dx', 'ip5']) > np.abs(tw2['dx', 'ip5']) * 4

# Check crab dispersion
xo.assert_allclose(tw1['dx_zeta', 'ip1'], -190e-6, atol=10e-6)
xo.assert_allclose(tw2['dx_zeta', 'ip1'], 190e-6, atol=10e-6)
xo.assert_allclose(tw1['dy_zeta', 'ip1'], 0, atol=5e-6)
xo.assert_allclose(tw2['dy_zeta', 'ip1'], 0, atol=5e-6)
xo.assert_allclose(tw1['dx_zeta', 'ip5'], 0, atol=5e-6)
xo.assert_allclose(tw2['dx_zeta', 'ip5'], 0, atol=5e-6)
xo.assert_allclose(tw1['dy_zeta', 'ip5'], -190e-6, atol=10e-6)
xo.assert_allclose(tw2['dy_zeta', 'ip5'], 190e-6, atol=10e-6)

with xt.line._temp_knobs(lhc, dict(on_crab1=0, on_crab5=0)):
    tw1_no_crab = lhc.b1.twiss()
    tw2_no_crab = lhc.b2.twiss(reverse=True)
xo.assert_allclose(tw1_no_crab['dx_zeta', 'ip1'], 0, atol=1e-7)
xo.assert_allclose(tw2_no_crab['dx_zeta', 'ip1'], 0, atol=1e-7)
xo.assert_allclose(tw1_no_crab['dy_zeta', 'ip1'], 0, atol=1e-7)
xo.assert_allclose(tw2_no_crab['dy_zeta', 'ip1'], 0, atol=1e-7)
xo.assert_allclose(tw1_no_crab['dx_zeta', 'ip5'], 0, atol=1e-7)
xo.assert_allclose(tw2_no_crab['dx_zeta', 'ip5'], 0, atol=1e-7)
xo.assert_allclose(tw1_no_crab['dy_zeta', 'ip5'], 0, atol=1e-7)
xo.assert_allclose(tw2_no_crab['dy_zeta', 'ip5'], 0, atol=1e-7)

# Check luminoisity in ip8 (set by leveling)
ll_ip8 = xt.lumi.luminosity_from_twiss(
    n_colliding_bunches=2572,
    num_particles_per_bunch=2.2e11,
    ip_name='ip8',
    nemitt_x=2.5e-6,
    nemitt_y=2.5e-6,
    sigma_z=0.076,
    twiss_b1=lhc.b1.twiss(reverse=False),
    twiss_b2=lhc.b2.twiss(reverse=False),
    crab=False)
xo.assert_allclose(ll_ip8, 2e33, rtol=0.05)

# Check separation plane orthogonal to crossing in ip8 (set by leveling)
x_diff_ip8 = tw1['x', 'ip8'] - tw2['x', 'ip8']
y_diff_ip8 = tw1['y', 'ip8'] - tw2['y', 'ip8']
px_diff_ip8 = tw1['px', 'ip8'] - tw2['px', 'ip8']
py_diff_ip8 = tw1['py', 'ip8'] - tw2['py', 'ip8']
xo.assert_allclose(x_diff_ip8*px_diff_ip8 + y_diff_ip8*py_diff_ip8, 0, atol=1e-12)

# Check separation in ip2
sigma_b1 = tw1.get_beam_covariance(nemitt_x=2.5e-6, nemitt_y=2.5e-6)
xo.assert_allclose((tw1['x', 'ip2'] - tw2['x', 'ip2'])/sigma_b1['sigma_x', 'ip2'],
                   5, rtol=0.1)
xo.assert_allclose(tw1['y', 'ip2'], 0, atol=5e-7)
xo.assert_allclose(tw2['y', 'ip2'], 0, atol=5e-7)

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
assert np.all(np.abs(global_chrom_coupling['b1']) < 6e-4)
assert np.all(np.abs(global_chrom_coupling['b2']) < 6e-4)

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

tw1_clean = lhc.b1.twiss4d(strengths=True)
tw2_clean = lhc.b2.twiss4d(strengths=True)

for tw in [tw1_clean, tw2_clean]:
    xo.assert_allclose(tw.rows['mq.*'].x, 0, atol=1e-9)
    xo.assert_allclose(tw.rows['mq.*'].px, 0, atol=1e-10)
    xo.assert_allclose(tw.rows['mq.*'].y, 0, atol=1e-10)
    xo.assert_allclose(tw.rows['mq.*'].py, 0, atol=1e-10)
    xo.assert_allclose(tw.qx, 62.31, atol=1e-6)
    xo.assert_allclose(tw.qy, 60.32, atol=1e-6)
    xo.assert_allclose(tw.dqx, 0, atol=1e-3)
    xo.assert_allclose(tw.dqy, 0, atol=1e-3)
    xo.assert_allclose(tw.c_minus, 0, atol=1e-4)
    xo.assert_allclose(tw.rows[['ip1', 'ip2', 'ip5', 'ip8']].betx,
                       [0.15, 10, 0.15, 1.5], rtol=1e-4)
    xo.assert_allclose(tw.k1sl, 0, atol=1e-12)
    xo.assert_allclose(tw.k2sl, 0, atol=1e-12)
    xo.assert_allclose(tw.k3sl, 0, atol=1e-12)
    xo.assert_allclose(tw.k4l, 0, atol=1e-12)
    xo.assert_allclose(tw.k4sl, 0, atol=1e-12)
    xo.assert_allclose(tw.k5l, 0, atol=1e-12)
    xo.assert_allclose(tw.k5sl, 0, atol=1e-12)

tw1_clean_6d = lhc.b1.twiss6d(strengths=True)
tw2_clean_6d = lhc.b2.twiss6d(strengths=True)

for tw in [tw1_clean_6d, tw2_clean_6d]:
    xo.assert_allclose(tw.rows['mq.*'].x, 0, atol=5e-8)
    xo.assert_allclose(tw.rows['mq.*'].px, 0, atol=1e-8)
    xo.assert_allclose(tw.rows['mq.*'].y, 0, atol=1e-10)
    xo.assert_allclose(tw.rows['mq.*'].py, 0, atol=1e-10)
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