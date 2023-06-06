from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from cpymad.madx import Madx
import xtrack as xt
import xfields as xf

import xmask as xm
import xmask.lhc as xmlhc
import yaml

from _complementary_run3_ions import (
    _config_ion_yaml_str, build_sequence, apply_optics, orbit_correction_config,
    check_optics_orbit_etc, _get_z_centroids, filling_scheme)

test_data_dir = Path(__file__).parent.parent / "test_data"

def test_lhc_ion_0_create_collider():
    # Read config file
    config = yaml.safe_load(_config_ion_yaml_str)
    config_mad_model = config['config_mad']

    # Make mad environment
    xm.make_mad_environment(links={
        "optics_runII": test_data_dir / 'lhc_ion/runII',
        "optics_runIII": test_data_dir / 'lhc_ion/runIII',
        }
    )

    # Start mad
    mad_b1b2 = Madx(command_log="mad_collider.log")
    mad_b4 = Madx(command_log="mad_b4.log")

    # Build sequences
    build_sequence(mad_b1b2, mylhcbeam=1)
    build_sequence(mad_b4, mylhcbeam=4)

    # Apply optics (only for b1b2, b4 will be generated from b1b2)
    apply_optics(mad_b1b2, optics_file=config_mad_model['optics_file'])

    # Build xsuite collider
    collider = xmlhc.build_xsuite_collider(
        sequence_b1=mad_b1b2.sequence.lhcb1,
        sequence_b2=mad_b1b2.sequence.lhcb2,
        sequence_b4=mad_b4.sequence.lhcb2,
        beam_config=config_mad_model['beam_config'],
        enable_imperfections=config_mad_model['enable_imperfections'],
        enable_knob_synthesis=config_mad_model['enable_knob_synthesis'],
        pars_for_imperfections=config_mad_model['pars_for_imperfections'],
        ver_lhc_run=config_mad_model['ver_lhc_run'],
        ver_hllhc_optics=config_mad_model['ver_hllhc_optics'])


    assert len(collider.lines.keys()) == 4

    assert collider.vars['nrj'] == 7e12

    for line_name in collider.lines.keys():
        pref = collider[line_name].particle_ref
        assert np.isclose(pref.q0, 82, rtol=1e-10, atol=0)
        assert np.isclose(pref.energy0[0], 5.74e14, rtol=1e-10, atol=0)
        assert np.isclose(pref.mass0, 193687272900.0, rtol=1e-10, atol=0)
        assert np.isclose(pref.gamma0[0], 2963.54, rtol=1e-6, atol=0)

    # Save to file
    collider.to_json('collider_lhc_ion_00.json')


def test_lhc_ion_1_install_beambeam():


    collider = xt.Multiline.from_json('collider_lhc_ion_00.json')

    collider.install_beambeam_interactions(
        clockwise_line='lhcb1',
        anticlockwise_line='lhcb2',
        ip_names=['ip1', 'ip2', 'ip5', 'ip8'],
        delay_at_ips_slots=[0, 891, 0, 2670],
        num_long_range_encounters_per_side={
            'ip1': 25, 'ip2': 20, 'ip5': 25, 'ip8': 20},
        num_slices_head_on=11,
        harmonic_number=35640,
        bunch_spacing_buckets=10,
        sigmaz=0.0824)

    collider.to_json('collider_lhc_ion_01.json')

    # Check integrity of the collider after installation

    collider_before_save = collider
    dct = collider.to_dict()
    collider = xt.Multiline.from_dict(dct)
    collider.build_trackers()

    assert collider._bb_config['dataframes']['clockwise'].shape == (
        collider_before_save._bb_config['dataframes']['clockwise'].shape)
    assert collider._bb_config['dataframes']['anticlockwise'].shape == (
        collider_before_save._bb_config['dataframes']['anticlockwise'].shape)

    assert (collider._bb_config['dataframes']['clockwise']['elementName'].iloc[50]
        == collider_before_save._bb_config['dataframes']['clockwise']['elementName'].iloc[50])
    assert (collider._bb_config['dataframes']['anticlockwise']['elementName'].iloc[50]
        == collider_before_save._bb_config['dataframes']['anticlockwise']['elementName'].iloc[50])

    # Put in some orbit
    knobs = dict(on_x1=250, on_x5=-200, on_disp=1)

    for kk, vv in knobs.items():
        collider.vars[kk] = vv

    tw1_b1 = collider['lhcb1'].twiss(method='4d')
    tw1_b2 = collider['lhcb2'].twiss(method='4d')

    collider_ref = xt.Multiline.from_json('collider_lhc_ion_00.json')

    collider_ref.build_trackers()

    for kk, vv in knobs.items():
        collider_ref.vars[kk] = vv

    tw0_b1 = collider_ref['lhcb1'].twiss(method='4d')
    tw0_b2 = collider_ref['lhcb2'].twiss(method='4d')

    assert np.isclose(tw1_b1.qx, tw0_b1.qx, atol=1e-7, rtol=0)
    assert np.isclose(tw1_b1.qy, tw0_b1.qy, atol=1e-7, rtol=0)
    assert np.isclose(tw1_b2.qx, tw0_b2.qx, atol=1e-7, rtol=0)
    assert np.isclose(tw1_b2.qy, tw0_b2.qy, atol=1e-7, rtol=0)

    assert np.isclose(tw1_b1.dqx, tw0_b1.dqx, atol=1e-3, rtol=0)
    assert np.isclose(tw1_b1.dqy, tw0_b1.dqy, atol=1e-3, rtol=0)
    assert np.isclose(tw1_b2.dqx, tw0_b2.dqx, atol=1e-3, rtol=0)
    assert np.isclose(tw1_b2.dqy, tw0_b2.dqy, atol=1e-3, rtol=0)

    for ipn in [1, 2, 3, 4, 5, 6, 7, 8]:
        assert np.isclose(tw1_b1['betx', f'ip{ipn}'], tw0_b1['betx', f'ip{ipn}'], rtol=1e-5, atol=0)
        assert np.isclose(tw1_b1['bety', f'ip{ipn}'], tw0_b1['bety', f'ip{ipn}'], rtol=1e-5, atol=0)
        assert np.isclose(tw1_b2['betx', f'ip{ipn}'], tw0_b2['betx', f'ip{ipn}'], rtol=1e-5, atol=0)
        assert np.isclose(tw1_b2['bety', f'ip{ipn}'], tw0_b2['bety', f'ip{ipn}'], rtol=1e-5, atol=0)

        assert np.isclose(tw1_b1['px', f'ip{ipn}'], tw0_b1['px', f'ip{ipn}'], rtol=1e-6, atol=0)
        assert np.isclose(tw1_b1['py', f'ip{ipn}'], tw0_b1['py', f'ip{ipn}'], rtol=1e-6, atol=0)
        assert np.isclose(tw1_b2['px', f'ip{ipn}'], tw0_b2['px', f'ip{ipn}'], rtol=1e-6, atol=0)
        assert np.isclose(tw1_b2['py', f'ip{ipn}'], tw0_b2['py', f'ip{ipn}'], rtol=1e-6, atol=0)

        assert np.isclose(tw1_b1['s', f'ip{ipn}'], tw0_b1['s', f'ip{ipn}'], rtol=1e-10, atol=0)
        assert np.isclose(tw1_b2['s', f'ip{ipn}'], tw0_b2['s', f'ip{ipn}'], rtol=1e-10, atol=0)


def test_lhc_ion_2_tuning():

    collider = xt.Multiline.from_json('collider_lhc_ion_01.json')

    # Read config file
    config = yaml.safe_load(_config_ion_yaml_str)
    conf_knobs_and_tuning = config['config_knobs_and_tuning']

    knob_settings = conf_knobs_and_tuning['knob_settings']
    tune_chorma_targets = conf_knobs_and_tuning
    knob_names_lines = conf_knobs_and_tuning['knob_names']

    # Set all knobs (crossing angles, dispersion correction, rf, crab cavities,
    # experimental magnets, etc.)
    for kk, vv in knob_settings.items():
        collider.vars[kk] = vv

    # Build trackers
    collider.build_trackers()

    # Check coupling knobs are responding
    collider.vars['cmrs.b1_sq'] = 1e-3
    collider.vars['cmis.b1_sq'] = 1e-3
    assert np.isclose(collider['lhcb1'].twiss().c_minus, 1.4e-3,
                      rtol=0, atol=2e-4)
    assert np.isclose(collider['lhcb2'].twiss().c_minus, 0,
                      rtol=0, atol=2e-4)
    collider.vars['cmrs.b1_sq'] = 0
    collider.vars['cmis.b1_sq'] = 0
    collider.vars['cmrs.b2_sq'] = 1e-3
    collider.vars['cmis.b2_sq'] = 1e-3
    assert np.isclose(collider['lhcb1'].twiss().c_minus, 0,
                        rtol=0, atol=2e-4)
    assert np.isclose(collider['lhcb2'].twiss().c_minus, 1.4e-3,
                        rtol=0, atol=2e-4)
    collider.vars['cmrs.b2_sq'] = 0
    collider.vars['cmis.b2_sq'] = 0

    # Introduce some coupling to check correction
    collider.vars['cmrs.b1_sq'] = 0.4e-3
    collider.vars['cmis.b1_sq'] = 0.7e-3
    collider.vars['cmrs.b2_sq'] = 0.5e-3
    collider.vars['cmis.b2_sq'] = 0.6e-3

    # Tunings
    for line_name in ['lhcb1', 'lhcb2']:

        knob_names = knob_names_lines[line_name]

        targets = {
            'qx': tune_chorma_targets['qx'][line_name],
            'qy': tune_chorma_targets['qy'][line_name],
            'dqx': tune_chorma_targets['dqx'][line_name],
            'dqy': tune_chorma_targets['dqy'][line_name],
        }

        xm.machine_tuning(line=collider[line_name],
            enable_closed_orbit_correction=True,
            enable_linear_coupling_correction=True,
            enable_tune_correction=True,
            enable_chromaticity_correction=True,
            knob_names=knob_names,
            targets=targets,
            line_co_ref=collider[line_name+'_co_ref'],
            co_corr_config=orbit_correction_config[line_name])

    collider.to_json('collider_lhc_ion_02.json')

    # Check optics, orbit, rf, etc.
    check_optics_orbit_etc(collider, line_names=['lhcb1', 'lhcb2'],
                           sep_h_ip1=1e-6, sep_h_ip2=1e-6,
                           sep_v_ip5=1e-6, sep_v_ip8=1e-6) # Setting in yaml file

def test_lhc_ion_3_leveling():

    # Load collider and build trackers
    collider = xt.Multiline.from_json('collider_lhc_ion_02.json')
    collider.build_trackers()

    config = yaml.safe_load(_config_ion_yaml_str)
    conf_knobs_and_tuning = config['config_knobs_and_tuning']
    config_lumi_leveling = config['config_lumi_leveling']
    config_beambeam = config['config_beambeam']

    xmlhc.luminosity_leveling(
        collider, config_lumi_leveling=config_lumi_leveling,
        config_beambeam=config_beambeam)

    # Re-match tunes, and chromaticities
    tune_chorma_targets = conf_knobs_and_tuning
    knob_names_lines = conf_knobs_and_tuning['knob_names']

    for line_name in ['lhcb1', 'lhcb2']:
        knob_names = knob_names_lines[line_name]
        targets = {
            'qx': tune_chorma_targets['qx'][line_name],
            'qy': tune_chorma_targets['qy'][line_name],
            'dqx': tune_chorma_targets['dqx'][line_name],
            'dqy': tune_chorma_targets['dqy'][line_name],
        }
        xm.machine_tuning(line=collider[line_name],
            enable_tune_correction=True, enable_chromaticity_correction=True,
            knob_names=knob_names, targets=targets)

    collider.to_json('collider_lhc_ion_03.json')

    # Checks
    import numpy as np
    tw = collider.twiss(lines=['lhcb1', 'lhcb2'])

    # Check luminosity in ip1
    ll_ip1 = xt.lumi.luminosity_from_twiss(
        n_colliding_bunches=1088,
        num_particles_per_bunch=180000000.0,
        ip_name='ip1',
        nemitt_x=1.65e-6,
        nemitt_y=1.65e-6,
        sigma_z=0.0824,
        twiss_b1=tw.lhcb1,
        twiss_b2=tw.lhcb2,
        crab=False)

    assert np.isclose(ll_ip1, 6.4e+27, rtol=1e-2, atol=0)

    # Check luminosity in ip2
    ll_ip2 = xt.lumi.luminosity_from_twiss(
        n_colliding_bunches=1088,
        num_particles_per_bunch=180000000.0,
        ip_name='ip2',
        nemitt_x=1.65e-6,
        nemitt_y=1.65e-6,
        sigma_z=0.0824,
        twiss_b1=tw.lhcb1,
        twiss_b2=tw.lhcb2,
        crab=False)

    assert np.isclose(ll_ip2, 6.4e+27, rtol=1e-2, atol=0)

    # Check luminosity in ip5
    ll_ip5 = xt.lumi.luminosity_from_twiss(
        n_colliding_bunches=1088,
        num_particles_per_bunch=180000000.0,
        ip_name='ip5',
        nemitt_x=1.65e-6,
        nemitt_y=1.65e-6,
        sigma_z=0.0824,
        twiss_b1=tw.lhcb1,
        twiss_b2=tw.lhcb2,
        crab=False)

    assert np.isclose(ll_ip5, 6.4e+27, rtol=1e-2, atol=0)

    # Check luminosity in ip8
    ll_ip8 = xt.lumi.luminosity_from_twiss(
        n_colliding_bunches=398,
        num_particles_per_bunch=180000000.0,
        ip_name='ip8',
        nemitt_x=1.65e-6,
        nemitt_y=1.65e-6,
        sigma_z=0.0824,
        twiss_b1=tw.lhcb1,
        twiss_b2=tw.lhcb2,
        crab=False)

    assert np.isclose(ll_ip8, 1e+27, rtol=1e-2, atol=0)

    # Check optics, orbit, rf, etc.
    check_optics_orbit_etc(collider, line_names=['lhcb1', 'lhcb2'],
                           # From lumi leveling
                           sep_h_ip1=0.00920e-3,
                           sep_h_ip2=0.01132e-3,
                           sep_v_ip5=0.00918e-3,
                           sep_v_ip8=0.01623e-3,
                           )

def test_lhc_ion_4_bb_config():

    collider = xt.Multiline.from_json('collider_lhc_ion_03.json')
    collider.build_trackers()

    collider.configure_beambeam_interactions(
        num_particles=180000000.0,
        nemitt_x=1.65e-6, nemitt_y=1.65e-6)

    collider.to_json('collider_lhc_ion_04.json')

    ip_bb_config= {
        'ip1': {'num_lr_per_side': 25},
        'ip2': {'num_lr_per_side': 20},
        'ip5': {'num_lr_per_side': 25},
        'ip8': {'num_lr_per_side': 20},
    }

    line_config = {
        'lhcb1': {'strong_beam': 'lhcb2', 'sorting': {'l': -1, 'r': 1}},
        'lhcb2': {'strong_beam': 'lhcb1', 'sorting': {'l': 1, 'r': -1}},
    }

    nemitt_x = 1.65e-6
    nemitt_y = 1.65e-6
    harmonic_number = 35640
    bunch_spacing_buckets = 10
    sigmaz = 0.0824
    num_slices_head_on = 11
    num_particles = 180000000.0
    qx_no_bb = {'lhcb1': 62.31, 'lhcb2': 62.31}
    qy_no_bb = {'lhcb1': 60.32, 'lhcb2': 60.32}

    for line_name in ['lhcb1', 'lhcb2']:

        print(f'Global check on line {line_name}')

        # Check that the number of lenses is correct
        df = collider[line_name].to_pandas()
        bblr_df = df[df['element_type'] == 'BeamBeamBiGaussian2D']
        bbho_df = df[df['element_type'] == 'BeamBeamBiGaussian3D']
        bb_df = pd.concat([bblr_df, bbho_df])

        assert (len(bblr_df) == 2 * sum(
            [ip_bb_config[ip]['num_lr_per_side'] for ip in ip_bb_config.keys()]))
        assert (len(bbho_df) == len(ip_bb_config.keys()) * num_slices_head_on)

        # Check that beam-beam scale knob works correctly
        collider.vars['beambeam_scale'] = 1
        for nn in bb_df.name.values:
            assert collider[line_name][nn].scale_strength == 1
        collider.vars['beambeam_scale'] = 0
        for nn in bb_df.name.values:
            assert collider[line_name][nn].scale_strength == 0
        collider.vars['beambeam_scale'] = 1
        for nn in bb_df.name.values:
            assert collider[line_name][nn].scale_strength == 1

        # Twiss with and without bb
        collider.vars['beambeam_scale'] = 1
        tw_bb_on = collider[line_name].twiss()
        collider.vars['beambeam_scale'] = 0
        tw_bb_off = collider[line_name].twiss()
        collider.vars['beambeam_scale'] = 1

        assert np.isclose(tw_bb_off.qx, qx_no_bb[line_name], rtol=0, atol=1e-4)
        assert np.isclose(tw_bb_off.qy, qy_no_bb[line_name], rtol=0, atol=1e-4)

        # Check that there is a tune shift of the order of 7e-4
        assert np.isclose(tw_bb_on.qx, qx_no_bb[line_name] - 7e-4, rtol=0, atol=4e-3)
        assert np.isclose(tw_bb_on.qy, qy_no_bb[line_name] - 7e-4, rtol=0, atol=4e-3)

        # Check that there is no effect on the orbit
        np.allclose(tw_bb_on.x, tw_bb_off.x, atol=1e-10, rtol=0)
        np.allclose(tw_bb_on.y, tw_bb_off.y, atol=1e-10, rtol=0)

    for name_weak, ip in product(['lhcb1', 'lhcb2'], ['ip1', 'ip2', 'ip5', 'ip8']):

        print(f'\n--> Checking {name_weak} {ip}\n')

        ip_n = int(ip[2])
        num_lr_per_side = ip_bb_config[ip]['num_lr_per_side']
        name_strong = line_config[name_weak]['strong_beam']
        sorting = line_config[name_weak]['sorting']

        # The bb lenses are setup based on the twiss taken with the bb off
        print('Twiss(es) (with bb off)')
        with xt._temp_knobs(collider, knobs={'beambeam_scale': 0}):
            tw_weak = collider[name_weak].twiss()
            tw_strong = collider[name_strong].twiss().reverse()

        # Survey starting from ip
        print('Survey(s) (starting from ip)')
        survey_weak = collider[name_weak].survey(element0=f'ip{ip_n}')
        survey_strong = collider[name_strong].survey(
                                            element0=f'ip{ip_n}').reverse()
        beta0_strong = collider[name_strong].particle_ref.beta0[0]
        gamma0_strong = collider[name_strong].particle_ref.gamma0[0]

        bunch_spacing_ds = (tw_weak.circumference / harmonic_number
                            * bunch_spacing_buckets)

        # Check lr encounters
        for side in ['l', 'r']:
            for iele in range(num_lr_per_side):
                nn_weak = f'bb_lr.{side}{ip_n}b{name_weak[-1]}_{iele+1:02d}'
                nn_strong = f'bb_lr.{side}{ip_n}b{name_strong[-1]}_{iele+1:02d}'

                assert nn_weak in tw_weak.name
                assert nn_strong in tw_strong.name

                ee_weak = collider[name_weak][nn_weak]

                assert isinstance(ee_weak, xf.BeamBeamBiGaussian2D)

                expected_sigma_x = np.sqrt(tw_strong['betx', nn_strong]
                                        * nemitt_x/beta0_strong/gamma0_strong)
                expected_sigma_y = np.sqrt(tw_strong['bety', nn_strong]
                                        * nemitt_y/beta0_strong/gamma0_strong)

                # Beam sizes
                assert np.isclose(ee_weak.other_beam_Sigma_11, expected_sigma_x**2,
                                atol=0, rtol=1e-5)
                assert np.isclose(ee_weak.other_beam_Sigma_33, expected_sigma_y**2,
                                atol=0, rtol=1e-5)

                # Check no coupling
                assert ee_weak.other_beam_Sigma_13 == 0

                # Orbit
                assert np.isclose(ee_weak.ref_shift_x, tw_weak['x', nn_weak],
                                rtol=0, atol=1e-4 * expected_sigma_x)
                assert np.isclose(ee_weak.ref_shift_y, tw_weak['y', nn_weak],
                                    rtol=0, atol=1e-4 * expected_sigma_y)

                # Separation
                assert np.isclose(ee_weak.other_beam_shift_x,
                    tw_strong['x', nn_strong] - tw_weak['x', nn_weak]
                    + survey_strong['X', nn_strong] - survey_weak['X', nn_weak],
                    rtol=0, atol=5e-4 * expected_sigma_x)

                assert np.isclose(ee_weak.other_beam_shift_y,
                    tw_strong['y', nn_strong] - tw_weak['y', nn_weak]
                    + survey_strong['Y', nn_strong] - survey_weak['Y', nn_weak],
                    rtol=0, atol=5e-4 * expected_sigma_y)

                # s position
                assert np.isclose(tw_weak['s', nn_weak] - tw_weak['s', f'ip{ip_n}'],
                                bunch_spacing_ds/2 * (iele+1) * sorting[side],
                                rtol=0, atol=10e-6)

                # Check intensity
                assert np.isclose(ee_weak.other_beam_num_particles, num_particles,
                                atol=0, rtol=1e-8)

                # Other checks
                assert ee_weak.min_sigma_diff < 1e-9
                assert ee_weak.min_sigma_diff > 0

                assert ee_weak.scale_strength == 1
                assert ee_weak.other_beam_q0 == 82

        # Check head on encounters

        # Quick check on _get_z_centroids
        assert np.isclose(np.mean(_get_z_centroids(100000, 5.)**2), 5**2,
                                rtol=0, atol=5e-4)
        assert np.isclose(np.mean(_get_z_centroids(100000, 5.)), 0,
                                rtol=0, atol=1e-10)

        z_centroids = _get_z_centroids(num_slices_head_on, sigmaz)
        assert len(z_centroids) == num_slices_head_on
        assert num_slices_head_on % 2 == 1

        # Measure crabbing angle
        z_crab_test = 0.01 # This is the z for the reversed strong beam (e.g. b2 and not b4)
        with xt._temp_knobs(collider, knobs={'beambeam_scale': 0}):
            tw_z_crab_plus = collider[name_strong].twiss(
                zeta0=-(z_crab_test), # This is the z for the physical strong beam (e.g. b4 and not b2)
                method='4d',
                freeze_longitudinal=True).reverse()
            tw_z_crab_minus = collider[name_strong].twiss(
                zeta0= -(-z_crab_test), # This is the z for the physical strong beam (e.g. b4 and not b2)
                method='4d',
                freeze_longitudinal=True).reverse()
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

            ee_weak = collider[name_weak][nn_weak]

            assert isinstance(ee_weak, xf.BeamBeamBiGaussian3D)
            assert ee_weak.num_slices_other_beam == 1
            assert ee_weak.slices_other_beam_zeta_center[0] == 0

            # s position
            expected_s = zz / 2
            assert np.isclose(tw_weak['s', nn_weak] - tw_weak['s', f'ip{ip_n}'],
                            expected_s, atol=10e-6, rtol=0)

            # Beam sizes
            expected_sigma_x = np.sqrt(tw_strong['betx', nn_strong]
                                    * nemitt_x/beta0_strong/gamma0_strong)
            expected_sigma_y = np.sqrt(tw_strong['bety', nn_strong]
                                    * nemitt_y/beta0_strong/gamma0_strong)

            assert np.isclose(ee_weak.slices_other_beam_Sigma_11[0],
                            expected_sigma_x**2,
                            atol=0, rtol=1e-5)
            assert np.isclose(ee_weak.slices_other_beam_Sigma_33[0],
                            expected_sigma_y**2,
                            atol=0, rtol=1e-5)

            expected_sigma_px = np.sqrt(tw_strong['gamx', nn_strong]
                                        * nemitt_x/beta0_strong/gamma0_strong)
            expected_sigma_py = np.sqrt(tw_strong['gamy', nn_strong]
                                        * nemitt_y/beta0_strong/gamma0_strong)
            assert np.isclose(ee_weak.slices_other_beam_Sigma_22[0],
                            expected_sigma_px**2,
                            atol=0, rtol=1e-4)
            assert np.isclose(ee_weak.slices_other_beam_Sigma_44[0],
                            expected_sigma_py**2,
                            atol=0, rtol=1e-4)

            expected_sigma_xpx = -(tw_strong['alfx', nn_strong]
                                    * nemitt_x / beta0_strong / gamma0_strong)
            expected_sigma_ypy = -(tw_strong['alfy', nn_strong]
                                    * nemitt_y / beta0_strong / gamma0_strong)
            assert np.isclose(ee_weak.slices_other_beam_Sigma_12[0],
                            expected_sigma_xpx,
                            atol=1e-12, rtol=5e-4)
            assert np.isclose(ee_weak.slices_other_beam_Sigma_34[0],
                            expected_sigma_ypy,
                            atol=1e-12, rtol=5e-4)

            # Assert no coupling
            assert ee_weak.slices_other_beam_Sigma_13[0] == 0
            assert ee_weak.slices_other_beam_Sigma_14[0] == 0
            assert ee_weak.slices_other_beam_Sigma_23[0] == 0
            assert ee_weak.slices_other_beam_Sigma_24[0] == 0

            # Orbit
            assert np.isclose(ee_weak.ref_shift_x, tw_weak['x', nn_weak],
                                rtol=0, atol=1e-4 * expected_sigma_x)
            assert np.isclose(ee_weak.ref_shift_px, tw_weak['px', nn_weak],
                                rtol=0, atol=1e-4 * expected_sigma_px)
            assert np.isclose(ee_weak.ref_shift_y, tw_weak['y', nn_weak],
                                rtol=0, atol=1e-4 * expected_sigma_y)
            assert np.isclose(ee_weak.ref_shift_py, tw_weak['py', nn_weak],
                                rtol=0, atol=1e-4 * expected_sigma_py)
            assert np.isclose(ee_weak.ref_shift_zeta, tw_weak['zeta', nn_weak],
                                rtol=0, atol=1e-9)
            assert np.isclose(ee_weak.ref_shift_pzeta,
                            tw_weak['ptau', nn_weak]/beta0_strong,
                            rtol=0, atol=1e-9)

            # Separation
            # for phi_crab definition, see Xsuite physics manual
            assert np.isclose(ee_weak.other_beam_shift_x,
                (tw_strong['x', nn_strong] - tw_weak['x', nn_weak]
                + survey_strong['X', nn_strong] - survey_weak['X', nn_weak]
                - phi_crab_x
                    * tw_strong.circumference / (2 * np.pi * harmonic_number)
                    * np.sin(2 * np.pi * zz
                            * harmonic_number / tw_strong.circumference)),
                rtol=0, atol=1e-6) # Not the cleanest, to be investigated

            assert np.isclose(ee_weak.other_beam_shift_y,
                (tw_strong['y', nn_strong] - tw_weak['y', nn_weak]
                + survey_strong['Y', nn_strong] - survey_weak['Y', nn_weak]
                - phi_crab_y
                    * tw_strong.circumference / (2 * np.pi * harmonic_number)
                    * np.sin(2 * np.pi * zz
                            * harmonic_number / tw_strong.circumference)),
                rtol=0, atol=1e-6) # Not the cleanest, to be investigated

            assert ee_weak.other_beam_shift_px == 0
            assert ee_weak.other_beam_shift_py == 0
            assert ee_weak.other_beam_shift_zeta == 0
            assert ee_weak.other_beam_shift_pzeta == 0

            # Check crossing angle
            # Assume that crossing is either in x or in y
            if np.abs(tw_weak['px', f'ip{ip_n}']) < 1e-6:
                # Vertical crossing
                assert np.isclose(ee_weak.alpha, np.pi/2, atol=5e-3, rtol=0)
                assert np.isclose(
                    2*ee_weak.phi,
                    tw_weak['py', f'ip{ip_n}'] - tw_strong['py', f'ip{ip_n}'],
                    atol=2e-7, rtol=0)
            else:
                # Horizontal crossing
                assert np.isclose(ee_weak.alpha,
                    (-52e-3 if ip_n==8 else 0)*{'lhcb1': 1, 'lhcb2': -1}[name_weak],
                    atol=5e-3, rtol=0)
                assert np.isclose(
                    2*ee_weak.phi,
                    tw_weak['px', f'ip{ip_n}'] - tw_strong['px', f'ip{ip_n}'],
                    atol=2e-7, rtol=0)

            # Check intensity
            assert np.isclose(ee_weak.slices_other_beam_num_particles[0],
                            num_particles/num_slices_head_on, atol=0, rtol=1e-8)

            # Other checks
            assert ee_weak.min_sigma_diff < 1e-9
            assert ee_weak.min_sigma_diff > 0

            assert ee_weak.threshold_singular < 1e-27
            assert ee_weak.threshold_singular > 0

            assert ee_weak._flag_beamstrahlung == 0

            assert ee_weak.scale_strength == 1
            assert ee_weak.other_beam_q0 == 82

            for nn in ['x', 'y', 'zeta', 'px', 'py', 'pzeta']:
                assert getattr(ee_weak, f'slices_other_beam_{nn}_center')[0] == 0


    # Check optics and orbit with bb off
    collider.vars['beambeam_scale'] = 0
    # Check optics, orbit, rf, etc.
    check_optics_orbit_etc(collider, line_names=['lhcb1', 'lhcb2'],
                           # From lumi leveling
                           sep_h_ip1=0.00920e-3,
                           sep_h_ip2=0.01132e-3,
                           sep_v_ip5=0.00918e-3,
                           sep_v_ip8=0.01623e-3,
                           )

def test_lhc_ion_5_filling_scheme():
    collider = xt.Multiline.from_json('collider_lhc_ion_04.json')
    collider.build_trackers()

    twb1 = collider.lhcb1.twiss()
    twb2 = collider.lhcb2.twiss()

    assert np.sum([collider.lhcb1[nn].scale_strength
                for nn in twb1.rows['bb_lr.r1b1_.*'].name]) == 25
    assert np.sum([collider.lhcb2[nn].scale_strength
                    for nn in twb2.rows['bb_lr.r1b2_.*'].name]) == 25
    assert np.sum([collider.lhcb1[nn].scale_strength
                    for nn in twb1.rows['bb_lr.r2b1_.*'].name]) == 20
    assert np.sum([collider.lhcb2[nn].scale_strength
                    for nn in twb2.rows['bb_lr.r2b2_.*'].name]) == 20
    assert np.sum([collider.lhcb1[nn].scale_strength
                    for nn in twb1.rows['bb_lr.r5b1_.*'].name]) == 25
    assert np.sum([collider.lhcb2[nn].scale_strength
                    for nn in twb2.rows['bb_lr.r5b2_.*'].name]) == 25
    assert np.sum([collider.lhcb1[nn].scale_strength
                    for nn in twb1.rows['bb_lr.r8b1_.*'].name]) == 20
    assert np.sum([collider.lhcb2[nn].scale_strength
                    for nn in twb2.rows['bb_lr.r8b2_.*'].name]) == 20

    collider.apply_filling_pattern(
        filling_pattern_cw=filling_scheme['beam1'],
        filling_pattern_acw=filling_scheme['beam2'],
        i_bunch_cw=488, i_bunch_acw=488)

    assert np.sum([collider.lhcb1[nn].scale_strength
                for nn in twb1.rows['bb_lr.r1b1_.*'].name]) == 11
    assert np.sum([collider.lhcb2[nn].scale_strength
                    for nn in twb2.rows['bb_lr.r1b2_.*'].name]) == 11
    assert np.sum([collider.lhcb1[nn].scale_strength
                    for nn in twb1.rows['bb_lr.r2b1_.*'].name]) == 9
    assert np.sum([collider.lhcb2[nn].scale_strength
                    for nn in twb2.rows['bb_lr.r2b2_.*'].name]) == 9
    assert np.sum([collider.lhcb1[nn].scale_strength
                    for nn in twb1.rows['bb_lr.r5b1_.*'].name]) == 11
    assert np.sum([collider.lhcb2[nn].scale_strength
                    for nn in twb2.rows['bb_lr.r5b2_.*'].name]) == 11
    assert np.sum([collider.lhcb1[nn].scale_strength
                    for nn in twb1.rows['bb_lr.r8b1_.*'].name]) == 9
    assert np.sum([collider.lhcb2[nn].scale_strength
                    for nn in twb2.rows['bb_lr.r8b2_.*'].name]) == 9