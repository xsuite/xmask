import xtrack as xt
import xmask as xm
import xmask.lhc as xmlhc
from pathlib import Path


test_data_dir = Path(__file__).parent.parent / "test_data"

def test_hllhc19():
    # Read config file
    with open(test_data_dir / 'hllhc19/config.yaml','r') as fid:
        config = xm.yaml.load(fid)

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


test_hllhc19() # Temporary, for development