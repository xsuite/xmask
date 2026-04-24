import xtrack as xt
import xmask as xm
import xobjects as xo
import xfields as xf
import xmask.lhc as xmlhc
from pathlib import Path
from itertools import product
import pytest
import numpy as np
import pandas as pd

test_data_dir = Path(__file__).parent.parent / "test_data"

@pytest.mark.parametrize("label", ['thin', 'thick'])
def test_hllhc19_run(label):

    # Read config file
    with open(test_data_dir / 'hllhc19/config.yaml','r') as fid:
        config = xm.yaml.load(fid)

    if label == 'thin':
        config['label'] = 'thin'
        config['lattice_file'] = 'lhc_thin.json'
        config['optics_file'] = 'opt_150_thin.madx'
    elif label == 'thick':
        config['label'] = 'thick'
        config['lattice_file'] = 'lhc.json'
        config['optics_file'] = 'opt_150.madx'

    # Load lattice
    lhc = xt.load(test_data_dir / f'hllhc19/{config["lattice_file"]}')

    # Clear default twiss settings
    lhc.b1.twiss_default.clear()
    lhc.b2.twiss_default.clear()

    # Load optics
    lhc.vars.load(test_data_dir / f'hllhc19/{config["optics_file"]}')

    # For legacy files:
    if 'particle_ref_b1' not in lhc.particles:
        if 'p0c' not in lhc.vars:
            lhc['p0c'] = 6.8e12
        lhc.new_particle(f'particle_ref_b1', p0c='p0c')
        lhc.new_particle(f'particle_ref_b2', p0c='p0c')
        lhc.b1.particle_ref = 'particle_ref_b1'
        lhc.b2.particle_ref = 'particle_ref_b2'

    assert 'particle_ref_b1' in lhc.particles
    assert 'particle_ref_b2' in lhc.particles
    assert lhc.b1.particle_ref.name == 'particle_ref_b1'
    assert lhc.b2.particle_ref.name == 'particle_ref_b2'

    # Define reference energy and rigidity variables
    lhc['energy0_b1'] = lhc.ref['particle_ref_b1'].energy0[0]
    lhc['energy0_b2'] = lhc.ref['particle_ref_b2'].energy0[0]
    lhc['brho0_b1'] = lhc.ref['particle_ref_b1'].rigidity0[0]
    lhc['brho0_b2'] = lhc.ref['particle_ref_b2'].rigidity0[0]

    # Define new knobs from yaml
    lhc.vars.default_to_zero = True # for knobs defined implicitly within expressions
    for knob_name, knob_expr in config['new_knobs'].items():
        lhc[knob_name] = knob_expr
    lhc.vars.default_to_zero = False

    # Attach orbit correction knobs to all dipole correctors
    lhc['on_corr_co'] = 1
    for kk in list(lhc.vars.keys()):
        if kk.startswith('acb'):
            lhc['corr_co_'+kk] = 0
            lhc.ref[kk] += (lhc.ref['corr_co_'+kk] * lhc.ref['on_corr_co'])

    # Cycle both beams
    lhc.b1.cycle('ip3')
    lhc.b2.cycle('ip3')

    # Install beam-beam lenses (inactive and not configured)
    config_bb = config['beam_beam']
    if config_bb['install_beam_beam']:
        lhc.install_beambeam_interactions(
            clockwise_line='b1',
            anticlockwise_line='b2',
            ip_names=['ip1', 'ip2', 'ip5', 'ip8'],
            delay_at_ips_slots=[0, 891, 0, 2670],
            num_long_range_encounters_per_side=
                config_bb['num_long_range_encounters_per_side'],
            num_slices_head_on=config_bb['num_slices_head_on'],
            harmonic_number=35640,
            bunch_spacing_buckets=config_bb['bunch_spacing_buckets'],
            sigmaz=config_bb['sigma_z'])

    # Prepare reference model for orbit correction
    lhc_co_ref = xmlhc.build_closed_orbit_reference(lhc)
    lhc_co_ref.to_json(f'lhc_co_ref_{config["label"]}.json')

    # Check that both lines twiss without errors
    twb1 = lhc.b1.twiss4d()
    twb2 = lhc.b2.twiss4d()


    #############################
    # Install multipolar errors #
    #############################

    apply_multipolar_errors_config = config['apply_multipolar_errors']

    if apply_multipolar_errors_config:
        # Read the configuration from the yaml
        err_conf = apply_multipolar_errors_config.pop('_config_')
        min_order = err_conf['min_order']
        max_order = err_conf['max_order']

        # Apply the errors
        for knob_name, json_file in apply_multipolar_errors_config.items():
            print(f'Applying multipolar errors from file to create knob {knob_name}')
            # Read the file
            multipole_errors = xt.json.load(test_data_dir / f'hllhc19/{json_file}')
            for line_name in ['b1', 'b2']:
                line = lhc[line_name]
                # Apply the errors in the line
                xm.set_multipole_errors_in_line(line, multipole_errors,
                                        min_order=min_order, max_order=max_order,
                                        error_knob_name=knob_name,
                                        append_order_to_knob_name=True)


    #####################################
    # Corrections for multipolar errors #
    #####################################


    # Force the knobs settings (on_error_... might be forced to 1 by the error,
    # installation and we want the user setting, it present to be applied on top of that)
    for knob_name, knob_value in config['knob_settings'].items():
        lhc[knob_name] = knob_value

    # Go to flat orbit
    vars_to_zero = config['knobs_to_zero_for_flat_orbit']
    tt_to_zero = lhc.vars.get_table(expr_obj=True).rows[vars_to_zero]
    lhc.set(tt_to_zero, 0)

    # Status of error knobs
    tt_err_knobs = lhc.vars.get_table().rows[r'on_error_.*']
    print("Error knobs in the environment:")
    tt_err_knobs.show()

    # Errors off to get reference twiss
    lhc.set(tt_err_knobs.name, 0)
    tw_b1 = lhc['b1'].twiss4d(reverse=False)
    tw_b2 = lhc['b2'].twiss4d(reverse=False)
    tw_b12 = {'b1': tw_b1, 'b2': tw_b2}

    # errors back on
    for nn in tt_err_knobs.name:
        lhc[nn] = tt_err_knobs['value', nn]

    # Local correction of IR15 multipole errors
    xmlhc.correct_ir_errors(lhc, twiss_b1=tw_b1, twiss_b2=tw_b2,
                            corrections=config['ir_corrections'])

    # Spool piece correctors (MCS, MC0, MCD)
    xmlhc.set_arc_spool_piece_correctors(lhc, twiss_b1=tw_b1, twiss_b2=tw_b2,
                                            use_mcs=True, use_mcd=True,
                                            use_mco=False) # dead circuits

    # k1s local + global correction (uses MQS)
    xmlhc.correct_k1s(lhc, twiss_b1=tw_b1, twiss_b2=tw_b2, feed_down=False)

    # k2s local + global correction (uses MSS)
    xmlhc.correct_k2s(lhc, twiss_b1=tw_b1, twiss_b2=tw_b2, feed_down=False)

    # Back to orbit with bumps
    for nn in tt_to_zero.name:
        expr_obj = tt_to_zero['expr_obj', nn]
        val = tt_to_zero['value', nn]
        if expr_obj is not None:
            lhc[nn] = expr_obj
        else:
            lhc[nn] = val

    # Set all knobs (crossing angles, dispersion correction, rf, crab cavities,
    # experimental magnets, etc.)
    for kk, vv in config['knob_settings'].items():
        lhc[kk] = vv

    # Reference model for orbit correction
    env_ref = xt.load(f'lhc_co_ref_{config["label"]}.json')

    # Tunings
    conf_tuning = config['tuning']
    optimizers = {}
    for line_name in ['b1', 'b2']:
        print()
        print('Working on line ', line_name)

        knob_names = conf_tuning['knob_names'][line_name]

        targets = {
            'qx': conf_tuning['qx'][line_name],
            'qy': conf_tuning['qy'][line_name],
            'dqx': conf_tuning['dqx'][line_name],
            'dqy': conf_tuning['dqy'][line_name],
        }

        optimizers[line_name] = xm.machine_tuning(line=lhc[line_name],
            enable_closed_orbit_correction=True,
            enable_linear_coupling_correction=True,
            enable_tune_correction=True,
            enable_chromaticity_correction=True,
            knob_names=knob_names,
            targets=targets,
            step_q_knob=conf_tuning['steps']['q_knob'],
            step_dq_knob=conf_tuning['steps']['dq_knob'],
            step_c_minus_knob=conf_tuning['steps']['c_minus_knob'],
            tol_tune=conf_tuning['tolerances']['tune'],
            tol_chromaticity=conf_tuning['tolerances']['chromaticity'],
            tol_c_minus=conf_tuning['tolerances']['c_minus'],
            line_co_ref=env_ref[line_name],
            co_corr_config=co_corr_config[line_name])

    ############
    # Leveling #
    ############

    config_lumi_leveling = config['lumi_leveling']
    config_beambeam = config['beam_beam']

    opts = xmlhc.luminosity_leveling(
        lhc, config_lumi_leveling=config_lumi_leveling,
        config_beambeam=config_beambeam)

    # Re-match tunes, and chromaticities
    conf_tuning = config['tuning']

    for line_name in ['b1', 'b2']:
        knob_names = conf_tuning['knob_names'][line_name]
        targets = {
            'qx': conf_tuning['qx'][line_name],
            'qy': conf_tuning['qy'][line_name],
            'dqx': conf_tuning['dqx'][line_name],
            'dqy': conf_tuning['dqy'][line_name],
        }
        xm.machine_tuning(line=lhc[line_name],
            enable_tune_correction=True, enable_chromaticity_correction=True,
            knob_names=knob_names, targets=targets)

    ##############################
    # Configure beam-beam lenses #
    ##############################

    print('Configuring beam-beam lenses...')
    lhc.configure_beambeam_interactions(
        num_particles=config_bb['num_particles_per_bunch'],
        nemitt_x=config_bb['nemitt_x'],
        nemitt_y=config_bb['nemitt_y'])

    lhc.to_json(f'lhc_{config["label"]}_tuned_and_leveled_bb_on.json')

@pytest.mark.parametrize("label", ['thin', 'thick'])
def test_hllhc19_check_config_and_tuning(label):

    # Load config
    with open(test_data_dir / f'hllhc19/config.yaml','r') as fid:
        config = xm.yaml.load(fid)

    lhc = xt.load(f'lhc_{label}_tuned_and_leveled_bb_on.json')

    ##########
    # Checks #
    ##########

    # Check that errors on a few magnet types are present,
    # and that the first two orders are zero (consistently with setup)
    assert np.max(np.abs(lhc['mb.a12r4.b2'].knl_rel)) > 10.
    assert np.all(lhc.get('mb.a12r4.b2').knl_rel[:2] == 0)
    assert np.max(np.abs(lhc['mb.a12r4.b2'].ksl_rel)) > 10.
    assert np.all(lhc.get('mb.a12r4.b2').ksl_rel[:2] == 0)

    assert np.max(np.abs(lhc['mq.12r4.b2'].knl_rel)) > 10.
    assert np.all(lhc.get('mq.12r4.b2').knl_rel[:2] == 0)
    assert np.max(np.abs(lhc['mq.12r4.b2'].ksl_rel)) > 10.
    assert np.all(lhc.get('mq.12r4.b2').ksl_rel[:2] == 0)

    tt_triplet_quads_15 = lhc.elements.get_table().rows['mqxf.*/b.*'].rows.match(element_type='Quadrupole')
    assert len(tt_triplet_quads_15) == 2 * 2 * 2 * 6 # 2 beams, 2 sides, 2 ips, 6 quads per triplet
    tt_d2_15 = lhc.elements.get_table().rows['mbrd.*4.*'].rows.match(element_type='RBend')
    assert len(tt_d2_15) == 2 * 2 * 2 # 2 beams, 2 sides, 2 ips
    for nn in list(tt_triplet_quads_15.name) + list(tt_d2_15.name):
        assert np.max(np.abs(lhc.get(nn).knl_rel)) > 10.
        assert np.all(lhc.get(nn).knl_rel[:2] == 0)
        assert np.max(np.abs(lhc.get(nn).ksl_rel)) > 10.
        assert np.all(lhc.get(nn).ksl_rel[:2] == 0)

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

    lhc['beambeam_scale'] = 0 # Beam beam off

    tw1 = lhc.b1.twiss()
    tw2 = lhc.b2.twiss(reverse=True)

    # Check global quantities
    xo.assert_allclose(tw1.qx, 62.31, atol=5e-5)
    xo.assert_allclose(tw1.qy, 60.32, atol=5e-5)
    xo.assert_allclose(tw2.qx, 62.31, atol=5e-5)
    xo.assert_allclose(tw2.qy, 60.32, atol=5e-5)
    xo.assert_allclose(tw1.dqx, 5, atol=0.1)
    xo.assert_allclose(tw1.dqy, 6, atol=0.1)
    xo.assert_allclose(tw2.dqx, 5, atol=0.1)
    xo.assert_allclose(tw2.dqy, 6, atol=0.1)
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

    # Check luminosity in ip8 (set by leveling)
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
    assert np.all(np.abs(global_chrom_coupling['b1']) < 5e-4)
    assert np.all(np.abs(global_chrom_coupling['b2']) < 5e-4)

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

    if label == 'thin':
        atol_x = 3e-7
        atol_px = 1e-8
    else:
        atol_x = 1e-9
        atol_px = 1e-10
    for tw in [tw1_clean, tw2_clean]:
        xo.assert_allclose(tw.rows['mq.*'].x, 0, atol=atol_x)
        xo.assert_allclose(tw.rows['mq.*'].px, 0, atol=atol_px)
        xo.assert_allclose(tw.rows['mq.*'].y, 0, atol=1e-10)
        xo.assert_allclose(tw.rows['mq.*'].py, 0, atol=1e-10)
        xo.assert_allclose(tw.qx, 62.31, atol=5e-5)
        xo.assert_allclose(tw.qy, 60.32, atol=5e-5)
        xo.assert_allclose(tw.dqx, 0, atol=5e-3)
        xo.assert_allclose(tw.dqy, 0, atol=5e-3)
        xo.assert_allclose(tw.c_minus, 0, atol=1e-4)
        xo.assert_allclose(tw.rows[['ip1', 'ip2', 'ip5', 'ip8']].betx,
                        [0.15, 10, 0.15, 1.5], rtol=3e-4)
        xo.assert_allclose(tw.k1sl, 0, atol=1e-12)
        xo.assert_allclose(tw.k2sl, 0, atol=1e-12)
        xo.assert_allclose(tw.k3sl, 0, atol=1e-12)
        xo.assert_allclose(tw.k4l, 0, atol=1e-12)
        xo.assert_allclose(tw.k4sl, 0, atol=1e-12)
        xo.assert_allclose(tw.k5l, 0, atol=1e-12)
        xo.assert_allclose(tw.k5sl, 0, atol=1e-12)

    tw1_clean_6d = lhc.b1.twiss6d(strengths=True)
    tw2_clean_6d = lhc.b2.twiss6d(strengths=True)
    if label == 'thin':
        atol_x = 3e-7
        atol_px = 1e-8
    else:
        atol_x = 3e-8
        atol_px = 1e-9
    for tw in [tw1_clean_6d, tw2_clean_6d]:
        xo.assert_allclose(tw.rows['mq.*'].x, 0, atol=atol_x)
        xo.assert_allclose(tw.rows['mq.*'].px, 0, atol=atol_px)
        xo.assert_allclose(tw.rows['mq.*'].y, 0, atol=1e-10)
        xo.assert_allclose(tw.rows['mq.*'].py, 0, atol=1e-10)
        xo.assert_allclose(tw.qx, 62.31, atol=5e-5)
        xo.assert_allclose(tw.qy, 60.32, atol=5e-5)
        xo.assert_allclose(tw.dqx, 0, atol=5e-3)
        xo.assert_allclose(tw.dqy, 0, atol=5e-3)
        xo.assert_allclose(tw.c_minus, 0, atol=1e-4)
        xo.assert_allclose(tw.qs, 0.0021243, atol=1e-5)
        xo.assert_allclose(tw.rows[['ip1', 'ip2', 'ip5', 'ip8']].betx,
                        [0.15, 10, 0.15, 1.5], rtol=3e-4)
        xo.assert_allclose(tw.k1sl, 0, atol=1e-12)
        xo.assert_allclose(tw.k2sl, 0, atol=1e-12)
        xo.assert_allclose(tw.k3sl, 0, atol=1e-12)
        xo.assert_allclose(tw.k4l, 0, atol=1e-12)
        xo.assert_allclose(tw.k4sl, 0, atol=1e-12)
        xo.assert_allclose(tw.k5l, 0, atol=1e-12)
        xo.assert_allclose(tw.k5sl, 0, atol=1e-12)

@pytest.mark.parametrize('label', ['thin', 'thick'])
def test_hllhc19_check_beam_beam(label):

    lhc = xt.load(f'lhc_{label}_tuned_and_leveled_bb_on.json')

    def _get_z_centroids(ho_slices, sigmaz):
        from scipy.stats import norm
        z_cuts = norm.ppf(
            np.linspace(0, 1, ho_slices + 1)[1:int((ho_slices + 1) / 2)]) * sigmaz
        z_centroids = []
        z_centroids.append(-sigmaz / np.sqrt(2*np.pi)
            * np.exp(-z_cuts[0]**2 / (2 * sigmaz * sigmaz)) * float(ho_slices))
        for ii,jj in zip(z_cuts[0:-1],z_cuts[1:]):
            z_centroids.append(-sigmaz / np.sqrt(2*np.pi)
                * (np.exp(-jj**2 / (2 * sigmaz * sigmaz))
                - np.exp(-ii**2 / (2 * sigmaz * sigmaz))
                ) * ho_slices)
        return np.array(z_centroids + [0] + [-ii for ii in z_centroids[-1::-1]])


    ip_bb_config= {
        'ip1': {'num_lr_per_side': 25},
        'ip2': {'num_lr_per_side': 20},
        'ip5': {'num_lr_per_side': 25},
        'ip8': {'num_lr_per_side': 20},
    }

    line_config = {
        'b1': {'strong_beam': 'b2', 'sorting': {'l': -1, 'r': 1}},
        'b2': {'strong_beam': 'b1', 'sorting': {'l': 1, 'r': -1}},
    }

    nemitt_x = 2.5e-6
    nemitt_y = 2.5e-6
    harmonic_number = 35640
    bunch_spacing_buckets = 10
    sigmaz = 0.076
    num_slices_head_on = 11
    num_particles = 2.2e11
    qx_no_bb = {'b1': 62.31, 'b2': 62.31}
    qy_no_bb = {'b1': 60.32, 'b2': 60.32}

    for name_weak, ip in product(['b1', 'b2'], ['ip1', 'ip2', 'ip5', 'ip8']):

        print(f'\n--> Checking {name_weak} {ip}\n')

        ip_n = int(ip[2])
        num_lr_per_side = ip_bb_config[ip]['num_lr_per_side']
        name_strong = line_config[name_weak]['strong_beam']
        sorting = line_config[name_weak]['sorting']

        # The bb lenses are setup based on the twiss taken with the bb off
        print('Twiss(es) (with bb off)')
        with xt._temp_knobs(lhc, knobs={'beambeam_scale': 0}):
            tw_weak = lhc[name_weak].twiss()
            tw_strong = lhc[name_strong].twiss().reverse()

        # Survey starting from ip
        print('Survey(s) (starting from ip)')
        survey_weak = lhc[name_weak].survey(element0=f'ip{ip_n}')
        survey_strong = lhc[name_strong].survey(
                                            element0=f'ip{ip_n}').reverse()
        beta0_strong = lhc[name_strong].particle_ref.beta0[0]
        gamma0_strong = lhc[name_strong].particle_ref.gamma0[0]

        bunch_spacing_ds = (tw_weak.line_length / harmonic_number
                            * bunch_spacing_buckets)

        # Check lr encounters
        for side in ['l', 'r']:
            for iele in range(num_lr_per_side):
                nn_weak = f'bb_lr.{side}{ip_n}b{name_weak[-1]}_{iele+1:02d}'
                nn_strong = f'bb_lr.{side}{ip_n}b{name_strong[-1]}_{iele+1:02d}'

                assert nn_weak in tw_weak.name
                assert nn_strong in tw_strong.name

                ee_weak = lhc[name_weak][nn_weak]

                assert isinstance(ee_weak, xf.BeamBeamBiGaussian2D)

                expected_sigma_x = np.sqrt(tw_strong['betx', nn_strong]
                                        * nemitt_x/beta0_strong/gamma0_strong)
                expected_sigma_y = np.sqrt(tw_strong['bety', nn_strong]
                                        * nemitt_y/beta0_strong/gamma0_strong)

                # Beam sizes
                xo.assert_allclose(np.sqrt(ee_weak.other_beam_Sigma_11), expected_sigma_x,
                                atol=0, rtol=1e-4)
                xo.assert_allclose(np.sqrt(ee_weak.other_beam_Sigma_33), expected_sigma_y,
                                atol=0, rtol=1e-4)

                # Check no coupling
                assert ee_weak.other_beam_Sigma_13 == 0

                # Orbit
                xo.assert_allclose(ee_weak.ref_shift_x, tw_weak['x', nn_weak],
                                rtol=0, atol=1e-4 * expected_sigma_x)
                xo.assert_allclose(ee_weak.ref_shift_y, tw_weak['y', nn_weak],
                                    rtol=0, atol=1e-4 * expected_sigma_y)

                # Separation
                xo.assert_allclose(ee_weak.other_beam_shift_x,
                    tw_strong['x', nn_strong] - tw_weak['x', nn_weak]
                    + survey_strong['X', nn_strong] - survey_weak['X', nn_weak],
                    rtol=0, atol=5e-4 * expected_sigma_x)

                xo.assert_allclose(ee_weak.other_beam_shift_y,
                    tw_strong['y', nn_strong] - tw_weak['y', nn_weak]
                    + survey_strong['Y', nn_strong] - survey_weak['Y', nn_weak],
                    rtol=0, atol=5e-4 * expected_sigma_y)

                # s position
                xo.assert_allclose(tw_weak['s', nn_weak] - tw_weak['s', f'ip{ip_n}'],
                                bunch_spacing_ds/2 * (iele+1) * sorting[side],
                                rtol=0, atol=10e-6)

                # Check intensity
                xo.assert_allclose(ee_weak.other_beam_num_particles, num_particles,
                                atol=0, rtol=1e-8)

                # Other checks
                assert ee_weak.min_sigma_diff < 1e-9
                assert ee_weak.min_sigma_diff > 0

                assert ee_weak.scale_strength == 1
                assert ee_weak.other_beam_q0 == 1

        # Check head on encounters

        # Quick check on _get_z_centroids
        xo.assert_allclose(np.mean(_get_z_centroids(100000, 5.)**2), 5**2,
                                rtol=0, atol=5e-4)
        xo.assert_allclose(np.mean(_get_z_centroids(100000, 5.)), 0,
                                rtol=0, atol=1e-10)

        z_centroids = _get_z_centroids(num_slices_head_on, sigmaz)
        assert len(z_centroids) == num_slices_head_on
        assert num_slices_head_on % 2 == 1

        # Measure crabbing angle
        z_crab_test = 0.01 # This is the z for the reversed strong beam (e.g. b2 and not b4)
        with xt._temp_knobs(lhc, knobs={'beambeam_scale': 0}):
            tw_z_crab_plus = lhc[name_strong].twiss(
                zeta0=-(z_crab_test), # This is the z for the physical strong beam (e.g. b4 and not b2)
                method='4d').reverse()
            tw_z_crab_minus = lhc[name_strong].twiss(
                zeta0= -(-z_crab_test), # This is the z for the physical strong beam (e.g. b4 and not b2)
                method='4d').reverse()
        phi_crab_x = -(
            (tw_z_crab_plus['x', f'ip{ip_n}'] - tw_z_crab_minus['x', f'ip{ip_n}'])
                / (2 * z_crab_test))
        phi_crab_y = -(
            (tw_z_crab_plus['y', f'ip{ip_n}'] - tw_z_crab_minus['y', f'ip{ip_n}'])
                / (2 * z_crab_test))

        for ii, zz in list(zip(range(-(num_slices_head_on - 1) // 2,
                            (num_slices_head_on - 1) // 2 + 1),
                        z_centroids)):

            if ii == 0:
                side = 'c'
            elif ii < 0:
                side = 'l' if sorting['l'] == -1 else 'r'
            else:
                side = 'r' if sorting['r'] == 1 else 'l'

            nn_weak = f'bb_ho.{side}{ip_n}b{name_weak[-1]}_{int(abs(ii)):02d}'
            nn_strong = f'bb_ho.{side}{ip_n}b{name_strong[-1]}_{int(abs(ii)):02d}'

            ee_weak = lhc[name_weak][nn_weak]

            assert isinstance(ee_weak, xf.BeamBeamBiGaussian3D)
            assert ee_weak.num_slices_other_beam == 1
            assert ee_weak.slices_other_beam_zeta_center[0] == 0

            # s position
            expected_s = zz / 2
            xo.assert_allclose(tw_weak['s', nn_weak] - tw_weak['s', f'ip{ip_n}'],
                            expected_s, atol=10e-6, rtol=0)

            # Beam sizes
            expected_sigma_x = np.sqrt(tw_strong['betx', nn_strong]
                                    * nemitt_x/beta0_strong/gamma0_strong)
            expected_sigma_y = np.sqrt(tw_strong['bety', nn_strong]
                                    * nemitt_y/beta0_strong/gamma0_strong)

            xo.assert_allclose(np.sqrt(ee_weak.slices_other_beam_Sigma_11[0]),
                            expected_sigma_x,
                            atol=0, rtol=1e-2)
            xo.assert_allclose(np.sqrt(ee_weak.slices_other_beam_Sigma_33[0]),
                            expected_sigma_y,
                            atol=0, rtol=1e-2)

            expected_sigma_px = np.sqrt(tw_strong['gamx', nn_strong]
                                        * nemitt_x/beta0_strong/gamma0_strong)
            expected_sigma_py = np.sqrt(tw_strong['gamy', nn_strong]
                                        * nemitt_y/beta0_strong/gamma0_strong)
            xo.assert_allclose(np.sqrt(ee_weak.slices_other_beam_Sigma_22[0]),
                            expected_sigma_px,
                            atol=0, rtol=1e-4)
            xo.assert_allclose(np.sqrt(ee_weak.slices_other_beam_Sigma_44[0]),
                            expected_sigma_py,
                            atol=0, rtol=1e-4)

            expected_sigma_xpx = -(tw_strong['alfx', nn_strong]
                                    * nemitt_x / beta0_strong / gamma0_strong)
            expected_sigma_ypy = -(tw_strong['alfy', nn_strong]
                                    * nemitt_y / beta0_strong / gamma0_strong)
            xo.assert_allclose(ee_weak.slices_other_beam_Sigma_12[0],
                            expected_sigma_xpx,
                            atol=1e-12, rtol=5e-4)
            xo.assert_allclose(ee_weak.slices_other_beam_Sigma_34[0],
                            expected_sigma_ypy,
                            atol=1e-12, rtol=5e-4)

            # Assert no coupling
            assert ee_weak.slices_other_beam_Sigma_13[0] == 0
            assert ee_weak.slices_other_beam_Sigma_14[0] == 0
            assert ee_weak.slices_other_beam_Sigma_23[0] == 0
            assert ee_weak.slices_other_beam_Sigma_24[0] == 0

            # Orbit
            xo.assert_allclose(ee_weak.ref_shift_x, tw_weak['x', nn_weak],
                                rtol=0, atol=1e-4 * expected_sigma_x)
            xo.assert_allclose(ee_weak.ref_shift_px, tw_weak['px', nn_weak],
                                rtol=0, atol=1e-4 * expected_sigma_px)
            xo.assert_allclose(ee_weak.ref_shift_y, tw_weak['y', nn_weak],
                                rtol=0, atol=1e-4 * expected_sigma_y)
            xo.assert_allclose(ee_weak.ref_shift_py, tw_weak['py', nn_weak],
                                rtol=0, atol=1e-4 * expected_sigma_py)
            xo.assert_allclose(ee_weak.ref_shift_zeta, tw_weak['zeta', nn_weak],
                                rtol=0, atol=1e-9)
            xo.assert_allclose(ee_weak.ref_shift_pzeta,
                            tw_weak['ptau', nn_weak]/beta0_strong,
                            rtol=0, atol=1e-9)

            # Separation
            # for phi_crab definition, see Xsuite physics manual
            xo.assert_allclose(ee_weak.other_beam_shift_x,
                (tw_strong['x', nn_strong] - tw_weak['x', nn_weak]
                + survey_strong['X', nn_strong] - survey_weak['X', nn_weak]
                - phi_crab_x
                    * tw_strong.line_length / (2 * np.pi * harmonic_number)
                    * np.sin(2 * np.pi * zz
                            * harmonic_number / tw_strong.line_length)),
                rtol=0, atol=1e-6) # Not the cleanest, to be investigated
            # print(f"nn_weak: {nn_weak}, nn_strong: {nn_strong}")
            # print(f"shift_x: {ee_weak.other_beam_shift_x}, expected shift_x: {(tw_strong['x', nn_strong] - tw_weak['x', nn_weak] + survey_strong['X', nn_strong] - survey_weak['X', nn_weak] - phi_crab_x * tw_strong.line_length / (2 * np.pi * harmonic_number) * np.sin(2 * np.pi * zz * harmonic_number / tw_strong.line_length))}")
            # print(f"x_strong: {tw_strong['x', nn_strong]}, x_weak: {tw_weak['x', nn_weak]}, survey_strong: {survey_strong['X', nn_strong]}, survey_weak: {survey_weak['X', nn_weak]}, phi_crab_x: {phi_crab_x}, zz: {zz}")

            xo.assert_allclose(ee_weak.other_beam_shift_y,
                (tw_strong['y', nn_strong] - tw_weak['y', nn_weak]
                + survey_strong['Y', nn_strong] - survey_weak['Y', nn_weak]
                - phi_crab_y
                    * tw_strong.line_length / (2 * np.pi * harmonic_number)
                    * np.sin(2 * np.pi * zz
                            * harmonic_number / tw_strong.line_length)),
                rtol=0, atol=1e-6) # Not the cleanest, to be investigated

            assert ee_weak.other_beam_shift_px == 0
            assert ee_weak.other_beam_shift_py == 0
            assert ee_weak.other_beam_shift_zeta == 0
            assert ee_weak.other_beam_shift_pzeta == 0

            # Check crossing plane orientation and crossing angle
            # General tilted-crossing relations from manual:
            #   alpha = atan(Delta p_y / Delta p_x) if |Delta p_x| >= |Delta p_y|
            #   alpha = pi/2 - atan(Delta p_x / Delta p_y) otherwise
            #   theta = 2 * phi
            delta_px = tw_weak['px', f'ip{ip_n}'] - tw_strong['px', f'ip{ip_n}']
            delta_py = tw_weak['py', f'ip{ip_n}'] - tw_strong['py', f'ip{ip_n}']
            theta_abs = np.hypot(delta_px, delta_py)

            if np.abs(delta_px) >= np.abs(delta_py):
                expected_alpha = np.arctan(delta_py / delta_px)
                theta_sign = np.sign(delta_px)
            else:
                expected_alpha = np.pi / 2 - np.arctan(delta_px / delta_py)
                theta_sign = np.sign(delta_py)

            expected_theta = theta_sign * theta_abs

            xo.assert_allclose(ee_weak.alpha, expected_alpha,
                                atol=5e-3, rtol=0)

            xo.assert_allclose(2 * ee_weak.phi, expected_theta,
                                atol=2e-7, rtol=0)

            # Enforce the signed decomposition of theta in the crossing plane.
            xo.assert_allclose(2 * ee_weak.phi * np.cos(ee_weak.alpha), delta_px,
                                atol=2e-7, rtol=0)
            xo.assert_allclose(2 * ee_weak.phi * np.sin(ee_weak.alpha), delta_py,
                                atol=2e-7, rtol=0)

            # check special cases (horizontal or vertical crossing) more explicitly
            if np.abs(tw_weak['px', f'ip{ip_n}']) < 1e-6:
                # Vertical crossing
                xo.assert_allclose(ee_weak.alpha, np.pi/2, atol=5e-3, rtol=0)
                xo.assert_allclose(
                    2 * ee_weak.phi,
                    tw_weak['py', f'ip{ip_n}'] - tw_strong['py', f'ip{ip_n}'],
                    atol=2e-7, rtol=0)
            elif np.abs(tw_weak['py', f'ip{ip_n}']) < 1e-6:
                # Horizontal crossing
                xo.assert_allclose(ee_weak.alpha, 0, atol=5e-3, rtol=0)
                xo.assert_allclose(
                    2 * ee_weak.phi,
                    tw_weak['px', f'ip{ip_n}'] - tw_strong['px', f'ip{ip_n}'],
                    atol=2e-7, rtol=0)

            # Check intensity
            xo.assert_allclose(ee_weak.slices_other_beam_num_particles[0],
                            num_particles/num_slices_head_on, atol=0, rtol=1e-8)

            # Other checks
            assert ee_weak.min_sigma_diff < 1e-9
            assert ee_weak.min_sigma_diff > 0

            assert ee_weak.threshold_singular < 1e-27
            assert ee_weak.threshold_singular > 0

            assert ee_weak.flag_beamstrahlung == 0

            assert ee_weak.scale_strength == 1
            assert ee_weak.other_beam_q0 == 1

            for nn in ['x', 'y', 'zeta', 'px', 'py', 'pzeta']:
                assert getattr(ee_weak, f'slices_other_beam_{nn}_center')[0] == 0

    for line_name in ['b1', 'b2']:

        print(f'Global check on line {line_name}')

        # Check that the number of lenses is correct
        df = lhc[line_name].to_pandas()
        bblr_df = df[df['element_type'] == 'BeamBeamBiGaussian2D']
        bbho_df = df[df['element_type'] == 'BeamBeamBiGaussian3D']
        bb_df = pd.concat([bblr_df, bbho_df])

        assert (len(bblr_df) == 2 * sum(
            [ip_bb_config[ip]['num_lr_per_side'] for ip in ip_bb_config.keys()]))
        assert (len(bbho_df) == len(ip_bb_config.keys()) * num_slices_head_on)

        # Check that beam-beam scale knob works correctly
        lhc.vars['beambeam_scale'] = 1
        for nn in bb_df.name.values:
            assert lhc[line_name][nn].scale_strength == 1
        lhc.vars['beambeam_scale'] = 0
        for nn in bb_df.name.values:
            assert lhc[line_name][nn].scale_strength == 0
        lhc.vars['beambeam_scale'] = 1
        for nn in bb_df.name.values:
            assert lhc[line_name][nn].scale_strength == 1

        # Twiss with and without bb
        lhc.vars['beambeam_scale'] = 1
        tw_bb_on = lhc[line_name].twiss()
        lhc.vars['beambeam_scale'] = 0
        tw_bb_off = lhc[line_name].twiss()
        lhc.vars['beambeam_scale'] = 1

        xo.assert_allclose(tw_bb_off.qx, qx_no_bb[line_name], rtol=0, atol=1e-4)
        xo.assert_allclose(tw_bb_off.qy, qy_no_bb[line_name], rtol=0, atol=1e-4)

        # Check that there is a tune shift of the order of 1.5e-2
        xo.assert_allclose(tw_bb_on.qx, qx_no_bb[line_name] - 1.5e-2, rtol=0, atol=5e-3)
        xo.assert_allclose(tw_bb_on.qy, qy_no_bb[line_name] - 1.5e-2, rtol=0, atol=5e-3)

        # Check that there is no effect on the orbit
        np.allclose(tw_bb_on.x, tw_bb_off.x, atol=1e-10, rtol=0)
        np.allclose(tw_bb_on.y, tw_bb_off.y, atol=1e-10, rtol=0)

        # Check size and position of the footprint
        fp_polar_with_rescale = lhc[line_name].get_footprint(
            nemitt_x=2.5e-6, nemitt_y=2.5e-6,
            r_range=[0.1, 5],
            linear_rescale_on_knobs=[
                xt.LinearRescale(knob_name='beambeam_scale', v0=0.0, dv=0.2)]
            )
        xo.assert_allclose(np.min(fp_polar_with_rescale.qx), 0.2941, atol=1e-3)
        xo.assert_allclose(np.max(fp_polar_with_rescale.qx), 0.3071, atol=1e-3)
        xo.assert_allclose(np.min(fp_polar_with_rescale.qy), 0.3036, atol=1e-3)
        xo.assert_allclose(np.max(fp_polar_with_rescale.qy), 0.3187, atol=1e-3)



# Closed orbit correction configuration for the machine tuning, needed above.
co_corr_config = {}
co_corr_config['b1'] = {
    'IR1 left': dict(
        ref_with_knobs={'on_corr_co': 0, 'on_disp': 0},
        start='e.ds.r8.b1',
        end='e.ds.l1.b1',
        vary=(
            'corr_co_acbh14.l1b1',
            'corr_co_acbh12.l1b1',
            'corr_co_acbv15.l1b1',
            'corr_co_acbv13.l1b1',
            ),
        targets=('e.ds.l1.b1',),
    ),
    'IR1 right': dict(
        ref_with_knobs={'on_corr_co': 0, 'on_disp': 0},
        start='s.ds.r1.b1',
        end='s.ds.l2.b1',
        vary=(
            'corr_co_acbh13.r1b1',
            'corr_co_acbh15.r1b1',
            'corr_co_acbv12.r1b1',
            'corr_co_acbv14.r1b1',
            ),
        targets=('s.ds.l2.b1',),
    ),
    'IR5 left': dict(
        ref_with_knobs={'on_corr_co': 0, 'on_disp': 0},
        start='e.ds.r4.b1',
        end='e.ds.l5.b1',
        vary=(
            'corr_co_acbh14.l5b1',
            'corr_co_acbh12.l5b1',
            'corr_co_acbv15.l5b1',
            'corr_co_acbv13.l5b1',
            ),
        targets=('e.ds.l5.b1',),
    ),
    'IR5 right': dict(
        ref_with_knobs={'on_corr_co': 0, 'on_disp': 0},
        start='s.ds.r5.b1',
        end='s.ds.l6.b1',
        vary=(
            'corr_co_acbh13.r5b1',
            'corr_co_acbh15.r5b1',
            'corr_co_acbv12.r5b1',
            'corr_co_acbv14.r5b1',
            ),
        targets=('s.ds.l6.b1',),
    ),
    'IP1': dict(
        ref_with_knobs={'on_corr_co': 0, 'on_disp': 0},
        start='e.ds.l1.b1',
        end='s.ds.r1.b1',
        vary=(
            'corr_co_acbch6.l1b1',
            'corr_co_acbcv5.l1b1',
            'corr_co_acbch5.r1b1',
            'corr_co_acbcv6.r1b1',
            'corr_co_acbyhs4.l1b1',
            'corr_co_acbyhs4.r1b1',
            'corr_co_acbyvs4.l1b1',
            'corr_co_acbyvs4.r1b1',
        ),
        targets=('ip1', 's.ds.r1.b1'),
    ),
    'IP2': dict(
        ref_with_knobs={'on_corr_co': 0, 'on_disp': 0},
        start='e.ds.l2.b1',
        end='s.ds.r2.b1',
        vary=(
            'corr_co_acbyhs5.l2b1',
            'corr_co_acbchs5.r2b1',
            'corr_co_acbyvs5.l2b1',
            'corr_co_acbcvs5.r2b1',
            'corr_co_acbyhs4.l2b1',
            'corr_co_acbyhs4.r2b1',
            'corr_co_acbyvs4.l2b1',
            'corr_co_acbyvs4.r2b1',
        ),
        targets=('ip2', 's.ds.r2.b1'),
    ),
    'IP5': dict(
        ref_with_knobs={'on_corr_co': 0, 'on_disp': 0},
        start='e.ds.l5.b1',
        end='s.ds.r5.b1',
        vary=(
            'corr_co_acbch6.l5b1',
            'corr_co_acbcv5.l5b1',
            'corr_co_acbch5.r5b1',
            'corr_co_acbcv6.r5b1',
            'corr_co_acbyhs4.l5b1',
            'corr_co_acbyhs4.r5b1',
            'corr_co_acbyvs4.l5b1',
            'corr_co_acbyvs4.r5b1',
        ),
        targets=('ip5', 's.ds.r5.b1'),
    ),
    'IP8': dict(
        ref_with_knobs={'on_corr_co': 0, 'on_disp': 0},
        start='e.ds.l8.b1',
        end='s.ds.r8.b1',
        vary=(
            'corr_co_acbch5.l8b1',
            'corr_co_acbyhs4.l8b1',
            'corr_co_acbyhs4.r8b1',
            'corr_co_acbyhs5.r8b1',
            'corr_co_acbcvs5.l8b1',
            'corr_co_acbyvs4.l8b1',
            'corr_co_acbyvs4.r8b1',
            'corr_co_acbyvs5.r8b1',
        ),
        targets=('ip8', 's.ds.r8.b1'),
    ),
}

co_corr_config['b2'] = {
    'IR1 left': dict(
        ref_with_knobs={'on_corr_co': 0, 'on_disp': 0},
        start='e.ds.l1.b2',
        end='e.ds.r8.b2',
        vary=(
            'corr_co_acbh13.l1b2',
            'corr_co_acbh15.l1b2',
            'corr_co_acbv12.l1b2',
            'corr_co_acbv14.l1b2',
            ),
        targets=('e.ds.r8.b2',),
    ),
    'IR1 right': dict(
        ref_with_knobs={'on_corr_co': 0, 'on_disp': 0},
        start='s.ds.l2.b2',
        end='s.ds.r1.b2',
        vary=(
            'corr_co_acbh12.r1b2',
            'corr_co_acbh14.r1b2',
            'corr_co_acbv13.r1b2',
            'corr_co_acbv15.r1b2',
            ),
        targets=('s.ds.r1.b2',),
    ),
    'IR5 left': dict(
        ref_with_knobs={'on_corr_co': 0, 'on_disp': 0},
        start='e.ds.l5.b2',
        end='e.ds.r4.b2',
        vary=(
            'corr_co_acbh13.l5b2',
            'corr_co_acbh15.l5b2',
            'corr_co_acbv12.l5b2',
            'corr_co_acbv14.l5b2',
            ),
        targets=('e.ds.r4.b2',),
    ),
    'IR5 right': dict(
        ref_with_knobs={'on_corr_co': 0, 'on_disp': 0},
        start='s.ds.l6.b2',
        end='s.ds.r5.b2',
        vary=(
            'corr_co_acbh12.r5b2',
            'corr_co_acbh14.r5b2',
            'corr_co_acbv13.r5b2',
            'corr_co_acbv15.r5b2',
            ),
        targets=('s.ds.r5.b2',),
    ),
    'IP1': dict(
        ref_with_knobs={'on_corr_co': 0, 'on_disp': 0},
        start='s.ds.r1.b2',
        end='e.ds.l1.b2',
        vary=(
            'corr_co_acbch6.r1b2',
            'corr_co_acbcv5.r1b2',
            'corr_co_acbch5.l1b2',
            'corr_co_acbcv6.l1b2',
            'corr_co_acbyhs4.l1b2',
            'corr_co_acbyhs4.r1b2',
            'corr_co_acbyvs4.l1b2',
            'corr_co_acbyvs4.r1b2',
        ),
        targets=('ip1', 'e.ds.l1.b2',),
    ),
    'IP2': dict(
        ref_with_knobs={'on_corr_co': 0, 'on_disp': 0},
        start='s.ds.r2.b2',
        end='e.ds.l2.b2',
        vary=(
            'corr_co_acbyhs5.l2b2',
            'corr_co_acbchs5.r2b2',
            'corr_co_acbyvs5.l2b2',
            'corr_co_acbcvs5.r2b2',
            'corr_co_acbyhs4.l2b2',
            'corr_co_acbyhs4.r2b2',
            'corr_co_acbyvs4.l2b2',
            'corr_co_acbyvs4.r2b2',
        ),
        targets=('ip2', 'e.ds.l2.b2'),
    ),
    'IP5': dict(
        ref_with_knobs={'on_corr_co': 0, 'on_disp': 0},
        start='s.ds.r5.b2',
        end='e.ds.l5.b2',
        vary=(
            'corr_co_acbch6.r5b2',
            'corr_co_acbcv5.r5b2',
            'corr_co_acbch5.l5b2',
            'corr_co_acbcv6.l5b2',
            'corr_co_acbyhs4.l5b2',
            'corr_co_acbyhs4.r5b2',
            'corr_co_acbyvs4.l5b2',
            'corr_co_acbyvs4.r5b2',
        ),
        targets=('ip5', 'e.ds.l5.b2',),
    ),
    'IP8': dict(
        ref_with_knobs={'on_corr_co': 0, 'on_disp': 0},
        start='s.ds.r8.b2',
        end='e.ds.l8.b2',
        vary=(
            'corr_co_acbchs5.l8b2',
            'corr_co_acbyhs5.r8b2',
            'corr_co_acbcvs5.l8b2',
            'corr_co_acbyvs5.r8b2',
            'corr_co_acbyhs4.l8b2',
            'corr_co_acbyhs4.r8b2',
            'corr_co_acbyvs4.l8b2',
            'corr_co_acbyvs4.r8b2',
        ),
        targets=('ip8', 'e.ds.l8.b2',),
    ),
}
