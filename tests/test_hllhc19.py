import xtrack as xt
import xmask as xm
import xobjects as xo
import xmask.lhc as xmlhc
from pathlib import Path
import numpy as np

test_data_dir = Path(__file__).parent.parent / "test_data"

def test_hllhc19():

    label = 'thin' # TODO: to be parametrized

    # Read config file
    with open(test_data_dir / 'hllhc19/config.yaml','r') as fid:
        config = xm.yaml.load(fid)

    config['label'] = 'thin'
    config['lattice_file'] = 'lhc_thin.json'
    config['optics_file'] = 'opt_150_thin.madx'

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

    ##########
    # Checks #
    ##########

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
        xo.assert_allclose(tw.qx, 62.31, atol=3e-6)
        xo.assert_allclose(tw.qy, 60.32, atol=3e-6)
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
        xo.assert_allclose(tw.rows['mq.*'].x, 0, atol=atol_x)
        xo.assert_allclose(tw.rows['mq.*'].px, 0, atol=atol_px)
        xo.assert_allclose(tw.rows['mq.*'].y, 0, atol=1e-10)
        xo.assert_allclose(tw.rows['mq.*'].py, 0, atol=1e-10)
        xo.assert_allclose(tw.qx, 62.31, atol=3e-6)
        xo.assert_allclose(tw.qy, 60.32, atol=3e-6)
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
