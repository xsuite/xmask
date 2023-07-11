import yaml
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from scipy.constants import c as clight
from scipy.constants import e as qe
from scipy.constants import epsilon_0

from cpymad.madx import Madx
import xtrack as xt
import xfields as xf
import xpart as xp

import xmask as xm
import xmask.lhc as xmlhc

from _complementary_hllhc14 import (build_sequence, apply_optics,
                                    check_optics_orbit_etc, orbit_correction_config,
                                    knob_settings_yaml_str, knob_names_yaml_str,
                                    tune_chroma_yaml_str, _get_z_centroids,
                                    leveling_yaml_str)

# We assume that the tests will be run in order. In case of issues we could use
# https://pypi.org/project/pytest-order/ to enforce the order.

test_data_dir = Path(__file__).parent.parent / "test_data"

def test_hllhc14_0_create_collider():
    # Make mad environment
    xm.make_mad_environment(links={
        'acc-models-lhc': str(test_data_dir / 'hllhc14')})

    # Start mad
    mad_b1b2 = Madx(command_log="mad_collider.log")
    mad_b4 = Madx(command_log="mad_b4.log")

    # Build sequences
    build_sequence(mad_b1b2, mylhcbeam=1)
    build_sequence(mad_b4, mylhcbeam=4)

    # Apply optics (only for b1b2, b4 will be generated from b1b2)
    apply_optics(mad_b1b2,
        optics_file="acc-models-lhc/round/opt_round_150_1500_thin.madx")

    # Build xsuite collider
    collider = xmlhc.build_xsuite_collider(
        sequence_b1=mad_b1b2.sequence.lhcb1,
        sequence_b2=mad_b1b2.sequence.lhcb2,
        sequence_b4=mad_b4.sequence.lhcb2,
        beam_config={'lhcb1':{'beam_energy_tot': 7000},
                     'lhcb2':{'beam_energy_tot': 7000}},
        enable_imperfections=False,
        enable_knob_synthesis='_mock_for_testing',
        rename_coupling_knobs=True,
        pars_for_imperfections={},
        ver_lhc_run=None,
        ver_hllhc_optics=1.4)

    assert len(collider.lines.keys()) == 4

    collider.to_json('collider_hllhc14_00.json')

def test_hllhc14_1_install_beambeam():

    collider = xt.Multiline.from_json('collider_hllhc14_00.json')

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
    sigmaz=0.076)

    collider.to_json('collider_hllhc14_01.json')

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

    collider_ref = xt.Multiline.from_json('collider_hllhc14_00.json')

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

        assert np.isclose(tw1_b1['px', f'ip{ipn}'], tw0_b1['px', f'ip{ipn}'], rtol=1e-9, atol=0)
        assert np.isclose(tw1_b1['py', f'ip{ipn}'], tw0_b1['py', f'ip{ipn}'], rtol=1e-9, atol=0)
        assert np.isclose(tw1_b2['px', f'ip{ipn}'], tw0_b2['px', f'ip{ipn}'], rtol=1e-9, atol=0)
        assert np.isclose(tw1_b2['py', f'ip{ipn}'], tw0_b2['py', f'ip{ipn}'], rtol=1e-9, atol=0)

        assert np.isclose(tw1_b1['s', f'ip{ipn}'], tw0_b1['s', f'ip{ipn}'], rtol=1e-10, atol=0)
        assert np.isclose(tw1_b2['s', f'ip{ipn}'], tw0_b2['s', f'ip{ipn}'], rtol=1e-10, atol=0)

def test_hllhc14_2_tuning():

    collider = xt.Multiline.from_json('collider_hllhc14_01.json')

    knob_settings = yaml.safe_load(knob_settings_yaml_str)
    tune_chorma_targets = yaml.safe_load(tune_chroma_yaml_str)
    knob_names_lines = yaml.safe_load(knob_names_yaml_str)

    # Set all knobs (crossing angles, dispersion correction, rf, crab cavities,
    # experimental magnets, etc.)
    for kk, vv in knob_settings.items():
        collider.vars[kk] = vv

    # Build trackers
    collider.build_trackers()

    # Check coupling knobs are responding
    collider.vars['c_minus_re_b1'] = 1e-3
    collider.vars['c_minus_im_b1'] = 1e-3
    assert np.isclose(collider['lhcb1'].twiss().c_minus, 1.4e-3,
                      rtol=0, atol=2e-4)
    assert np.isclose(collider['lhcb2'].twiss().c_minus, 0,
                      rtol=0, atol=2e-4)
    collider.vars['c_minus_re_b1'] = 0
    collider.vars['c_minus_im_b1'] = 0
    collider.vars['c_minus_re_b2'] = 1e-3
    collider.vars['c_minus_im_b2'] = 1e-3
    assert np.isclose(collider['lhcb1'].twiss().c_minus, 0,
                        rtol=0, atol=2e-4)
    assert np.isclose(collider['lhcb2'].twiss().c_minus, 1.4e-3,
                        rtol=0, atol=2e-4)
    collider.vars['c_minus_re_b2'] = 0
    collider.vars['c_minus_im_b2'] = 0

    # Introduce some coupling to check correction
    collider.vars['c_minus_re_b1'] = 0.4e-3
    collider.vars['c_minus_im_b1'] = 0.7e-3
    collider.vars['c_minus_re_b2'] = 0.5e-3
    collider.vars['c_minus_im_b2'] = 0.6e-3

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

    collider.to_json('collider_hllhc14_02.json')

    # Check optics, orbit, rf, etc.
    check_optics_orbit_etc(collider, line_names=['lhcb1', 'lhcb2'],
                           sep_h_ip2=-0.138e-3, sep_v_ip8=-0.043e-3) # Setting in yaml file

def test_hllhc14_3_level_ip2_ip8():

    # Load collider and build trackers
    collider = xt.Multiline.from_json('collider_hllhc14_02.json')
    collider.build_trackers()

    config = yaml.safe_load(leveling_yaml_str)
    config_lumi_leveling = config['config_lumi_leveling']
    config_beambeam = config['config_beambeam']

    xmlhc.luminosity_leveling(
        collider, config_lumi_leveling=config_lumi_leveling,
        config_beambeam=config_beambeam)

    # Re-match tunes, and chromaticities
    tune_chorma_targets = yaml.safe_load(tune_chroma_yaml_str)
    knob_names_lines = yaml.safe_load(knob_names_yaml_str)

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

    collider.to_json('collider_hllhc14_03.json')

    # Checks
    import numpy as np
    tw = collider.twiss(lines=['lhcb1', 'lhcb2'])

    # Check luminosity in ip8
    ll_ip8 = xt.lumi.luminosity_from_twiss(
        n_colliding_bunches=2572,
        num_particles_per_bunch=2.2e11,
        ip_name='ip8',
        nemitt_x=2.5e-6,
        nemitt_y=2.5e-6,
        sigma_z=0.076,
        twiss_b1=tw.lhcb1,
        twiss_b2=tw.lhcb2,
        crab=False)

    assert np.isclose(ll_ip8, 2e33, rtol=1e-2, atol=0)

    # Check separation in ip2
    mean_betx = np.sqrt(tw['lhcb1']['betx', 'ip2']
                    *tw['lhcb2']['betx', 'ip2'])
    gamma0 = tw['lhcb1'].particle_on_co.gamma0[0]
    beta0 = tw['lhcb1'].particle_on_co.beta0[0]
    sigmax = np.sqrt(2.5e-6 * mean_betx /gamma0 / beta0)

    assert np.isclose(collider.vars['on_sep2']._value/1000,
                      5 * sigmax / 2, rtol=1e-3, atol=0)

    # Check optics, orbit, rf, etc.
    check_optics_orbit_etc(collider, line_names=['lhcb1', 'lhcb2'],
                           # From lumi leveling
                           sep_h_ip2=-0.00014330344100935583, # checked against normalized sep
                           sep_v_ip8=-3.434909687327809e-05, # checked against lumi
                           )

def test_hllhc14_4_bb_config():

    collider = xt.Multiline.from_json('collider_hllhc14_03.json')
    collider.build_trackers()

    collider.configure_beambeam_interactions(
        num_particles=2.2e11,
        nemitt_x=2e-6, nemitt_y=3e-6)

    collider.to_json('collider_hllhc14_04.json')

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

    nemitt_x = 2e-6
    nemitt_y = 3e-6
    harmonic_number = 35640
    bunch_spacing_buckets = 10
    sigmaz = 0.076
    num_slices_head_on = 11
    num_particles = 2.2e11
    qx_no_bb = {'lhcb1': 62.31, 'lhcb2': 62.315}
    qy_no_bb = {'lhcb1': 60.32, 'lhcb2': 60.325}

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

        # Check that there is a tune shift of the order of 1.5e-2
        assert np.isclose(tw_bb_on.qx, qx_no_bb[line_name] - 1.5e-2, rtol=0, atol=5e-3)
        assert np.isclose(tw_bb_on.qy, qy_no_bb[line_name] - 1.5e-2, rtol=0, atol=5e-3)

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
                assert ee_weak.other_beam_q0 == 1

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
                    (-15e-3 if ip_n==8 else 0)*{'lhcb1': 1, 'lhcb2': -1}[name_weak],
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
            assert ee_weak.other_beam_q0 == 1

            for nn in ['x', 'y', 'zeta', 'px', 'py', 'pzeta']:
                assert getattr(ee_weak, f'slices_other_beam_{nn}_center')[0] == 0


    # Check optics and orbit with bb off
    collider.vars['beambeam_scale'] = 0
    check_optics_orbit_etc(collider, line_names=['lhcb1', 'lhcb2'],
                           # From lumi leveling
                           sep_h_ip2=-0.00014330344100935583, # checked against normalized sep
                           sep_v_ip8=-3.43490968732878e-05, # checked against lumi
                           )

def test_stress_co_correction_and_lumi_leveling():

    collider = xt.Multiline.from_json('collider_hllhc14_02.json')
    collider.build_trackers()

    num_colliding_bunches = 2808
    num_particles_per_bunch = 1.15e11
    nemitt_x = 3.75e-6
    nemitt_y = 3.75e-6
    sigma_z = 0.0755
    beta0_b1 = collider.lhcb1.particle_ref.beta0[0]
    f_rev=1/(collider.lhcb1.get_length() /(beta0_b1 * clight))

    # Move to external vertical crossing
    collider.vars['phi_ir8'] = 90.

    tw_before_errors = collider.twiss(lines=['lhcb1', 'lhcb2'])

    # Add errors
    for line_name in ['lhcb1', 'lhcb2']:
        collider[line_name]['mqxb.a2r8..5'].knl[0] = 1e-5
        collider[line_name]['mqxb.a2l8..5'].knl[0] = -0.7e-5
        collider[line_name]['mqxb.a2r8..5'].ksl[0] = -1.3e-5
        collider[line_name]['mqxb.a2l8..5'].ksl[0] = 0.9e-5

        collider[line_name]['mqxb.a2r8..5'].knl[1] = collider[line_name]['mqxb.a2r8..4'].knl[1] * 1.3
        collider[line_name]['mqxb.a2l8..5'].knl[1] = collider[line_name]['mqxb.a2l8..4'].knl[1] * 1.3
    collider.lhcb1['mqy.a4l8.b1..1'].knl[1] = collider.lhcb1['mqy.a4l8.b1..2'].knl[1] * 0.7
    collider.lhcb1['mqy.a4r8.b1..1'].knl[1] = collider.lhcb1['mqy.a4r8.b1..2'].knl[1] * 1.2
    collider.lhcb2['mqy.a4l8.b2..1'].knl[1] = collider.lhcb2['mqy.a4l8.b2..2'].knl[1] * 1.1
    collider.lhcb2['mqy.a4r8.b2..1'].knl[1] = collider.lhcb2['mqy.a4r8.b2..2'].knl[1] * 1.1

    tw_after_errors = collider.twiss(lines=['lhcb1', 'lhcb2'])

    # Correct orbit
    for line_name in ['lhcb1', 'lhcb2']:
        xm.machine_tuning(line=collider[line_name],
            enable_closed_orbit_correction=True,
            enable_linear_coupling_correction=False,
            enable_tune_correction=False,
            enable_chromaticity_correction=False,
            knob_names=[],
            targets=None,
            line_co_ref=collider[line_name+'_co_ref'],
            co_corr_config=orbit_correction_config[line_name])

    tw_after_orbit_correction = collider.twiss(lines=['lhcb1', 'lhcb2'])

    print(f'Knobs before matching: on_sep8h = {collider.vars["on_sep8h"]._value} '
            f'on_sep8v = {collider.vars["on_sep8v"]._value}')

    # Correction assuming ideal behavior of the knobs
    knob_values_before_ideal_matching = {
        'on_sep8h': collider.vars['on_sep8h']._value,
        'on_sep8v': collider.vars['on_sep8v']._value,
    }

    # Lumi leveling assuming ideal behavior of the knobs
    collider.match(
        solver_options={ # Standard jacobian settings not sufficient
                         #(fsolve makes it in less iterations)
            'n_bisections': 3, 'min_step': 0, 'n_steps_max': 200},
        ele_start=['e.ds.l8.b1', 's.ds.r8.b2'],
        ele_stop=['s.ds.r8.b1', 'e.ds.l8.b2'],
        twiss_init='preserve',
        lines=['lhcb1', 'lhcb2'],
        vary=[
            # Knobs to control the separation
            xt.Vary('on_sep8h', step=1e-4),
            xt.Vary('on_sep8v', step=1e-4),
        ],
        targets=[
            xt.TargetLuminosity(ip_name='ip8',
                                    luminosity=2e14,
                                    tol=1e12,
                                    f_rev=1/(collider.lhcb1.get_length() /(beta0_b1 * clight)),
                                    num_colliding_bunches=num_colliding_bunches,
                                    num_particles_per_bunch=num_particles_per_bunch,
                                    nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                                    sigma_z=sigma_z, crab=False),
            xt.TargetSeparationOrthogonalToCrossing(ip_name='ip8'),
        ],
    )

    tw_after_ideal_lumi_matching = collider.twiss(lines=['lhcb1', 'lhcb2'])

    # Reset knobs
    collider.vars['on_sep8h'] = knob_values_before_ideal_matching['on_sep8h']
    collider.vars['on_sep8v'] = knob_values_before_ideal_matching['on_sep8v']

    # Lumi leveling with orbit correction
    collider.match(
        solver_options={ # Standard jacobian settings not sufficient
                         #(fsolve makes it in less iterations)
            'n_bisections': 3, 'min_step': 0, 'n_steps_max': 200},
        lines=['lhcb1', 'lhcb2'],
        ele_start=['e.ds.l8.b1', 's.ds.r8.b2'],
        ele_stop=['s.ds.r8.b1', 'e.ds.l8.b2'],
        twiss_init='preserve',
        targets=[
            xt.TargetLuminosity(
                ip_name='ip8', luminosity=2e14, tol=1e12, f_rev=f_rev,
                num_colliding_bunches=num_colliding_bunches,
                num_particles_per_bunch=num_particles_per_bunch,
                nemitt_x=nemitt_x, nemitt_y=nemitt_y, sigma_z=sigma_z, crab=False),
            xt.TargetSeparationOrthogonalToCrossing(ip_name='ip8'),
            # Preserve crossing angle
            xt.TargetList(['px', 'py'], at='ip8', line='lhcb1', value='preserve', tol=1e-7, scale=1e3),
            xt.TargetList(['px', 'py'], at='ip8', line='lhcb2', value='preserve', tol=1e-7, scale=1e3),
            # Close the bumps
            xt.TargetList(['x', 'y'], at='s.ds.r8.b1', line='lhcb1', value='preserve', tol=1e-5, scale=1),
            xt.TargetList(['px', 'py'], at='s.ds.r8.b1', line='lhcb1', value='preserve', tol=1e-5, scale=1e3),
            xt.TargetList(['x', 'y'], at='e.ds.l8.b2', line='lhcb2', value='preserve', tol=1e-5, scale=1),
            xt.TargetList(['px', 'py'], at='e.ds.l8.b2', line='lhcb2', value='preserve', tol=1e-5, scale=1e3),
            ],
        vary=[
            xt.VaryList(['on_sep8h', 'on_sep8v'], step=1e-4), # to control separation
            xt.VaryList([
                # correctors to control the crossing angles
                'corr_co_acbyvs4.l8b1', 'corr_co_acbyhs4.l8b1',
                'corr_co_acbyvs4.r8b2', 'corr_co_acbyhs4.r8b2',
                # correctors to close the bumps
                'corr_co_acbyvs4.l8b2', 'corr_co_acbyhs4.l8b2',
                'corr_co_acbyvs4.r8b1', 'corr_co_acbyhs4.r8b1',
                'corr_co_acbcvs5.l8b2', 'corr_co_acbchs5.l8b2',
                'corr_co_acbyvs5.r8b1', 'corr_co_acbyhs5.r8b1'],
                step=1e-7),
        ],
    )

    print (f'Knobs after matching: on_sep8h = {collider.vars["on_sep8h"]._value} '
            f'on_sep8v = {collider.vars["on_sep8v"]._value}')

    tw_after_full_match = collider.twiss(lines=['lhcb1', 'lhcb2'])

    print(f'Before ideal matching: px = {tw_after_orbit_correction["lhcb1"]["px", "ip8"]:.3e} ')
    print(f'After ideal matching:  px = {tw_after_ideal_lumi_matching["lhcb1"]["px", "ip8"]:.3e} ')
    print(f'After full matching:   px = {tw_after_full_match["lhcb1"]["px", "ip8"]:.3e} ')
    print(f'Before ideal matching: py = {tw_after_orbit_correction["lhcb1"]["py", "ip8"]:.3e} ')
    print(f'After ideal matching:  py = {tw_after_ideal_lumi_matching["lhcb1"]["py", "ip8"]:.3e} ')
    print(f'After full matching:   py = {tw_after_full_match["lhcb1"]["py", "ip8"]:.3e} ')

    for place in ['ip1', 'ip8']:
        # Check that the errors are perturbing the crossing angles
        assert np.abs(tw_after_errors.lhcb1['px', place] - tw_before_errors.lhcb1['px', place]) > 10e-6
        assert np.abs(tw_after_errors.lhcb2['px', place] - tw_before_errors.lhcb2['px', place]) > 10e-6
        assert np.abs(tw_after_errors.lhcb1['py', place] - tw_before_errors.lhcb1['py', place]) > 10e-6
        assert np.abs(tw_after_errors.lhcb2['py', place] - tw_before_errors.lhcb2['py', place]) > 10e-6

        # Check that the orbit correction is restoring the crossing angles
        assert np.isclose(tw_after_orbit_correction.lhcb1['px', place],
                            tw_before_errors.lhcb1['px', place], atol=1e-6, rtol=0)
        assert np.isclose(tw_after_orbit_correction.lhcb2['px', place],
                            tw_before_errors.lhcb2['px', place], atol=1e-6, rtol=0)
        assert np.isclose(tw_after_orbit_correction.lhcb1['py', place],
                            tw_before_errors.lhcb1['py', place], atol=1e-6, rtol=0)
        assert np.isclose(tw_after_orbit_correction.lhcb2['py', place],
                            tw_before_errors.lhcb2['py', place], atol=1e-6, rtol=0)

        # Check that the ideal lumi matching is perturbing the crossing angles
        assert np.abs(tw_after_ideal_lumi_matching.lhcb1['px', place] - tw_before_errors.lhcb1['px', place]) > 1e-6
        assert np.abs(tw_after_ideal_lumi_matching.lhcb2['px', place] - tw_before_errors.lhcb2['px', place]) > 1e-6
        assert np.abs(tw_after_ideal_lumi_matching.lhcb1['py', place] - tw_before_errors.lhcb1['py', place]) > 1e-6
        assert np.abs(tw_after_ideal_lumi_matching.lhcb2['py', place] - tw_before_errors.lhcb2['py', place]) > 1e-6

        # Check that the full matching is preserving the crossing angles
        assert np.isclose(tw_after_full_match.lhcb1['px', place],
                            tw_before_errors.lhcb1['px', place], atol=1e-7, rtol=0)
        assert np.isclose(tw_after_full_match.lhcb2['px', place],
                            tw_before_errors.lhcb2['px', place], atol=1e-7, rtol=0)
        assert np.isclose(tw_after_full_match.lhcb1['py', place],
                            tw_before_errors.lhcb1['py', place], atol=1e-7, rtol=0)
        assert np.isclose(tw_after_full_match.lhcb2['py', place],
                            tw_before_errors.lhcb2['py', place], atol=1e-7, rtol=0)


    ll_after_match = xt.lumi.luminosity_from_twiss(
        n_colliding_bunches=num_colliding_bunches,
        num_particles_per_bunch=num_particles_per_bunch,
        ip_name='ip8',
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
        sigma_z=sigma_z,
        twiss_b1=tw_after_full_match['lhcb1'],
        twiss_b2=tw_after_full_match['lhcb2'],
        crab=False)

    assert np.isclose(ll_after_match, 2e14, rtol=1e-2, atol=0)

    # Check orthogonality
    tw_b1 = tw_after_full_match['lhcb1']
    tw_b4 = tw_after_full_match['lhcb2']
    tw_b2 = tw_b4.reverse()

    diff_px = tw_b1['px', 'ip8'] - tw_b2['px', 'ip8']
    diff_py = tw_b1['py', 'ip8'] - tw_b2['py', 'ip8']
    diff_x = tw_b1['x', 'ip8'] - tw_b2['x', 'ip8']
    diff_y = tw_b1['y', 'ip8'] - tw_b2['y', 'ip8']

    dpx_norm = diff_px / np.sqrt(diff_px**2 + diff_py**2)
    dpy_norm = diff_py / np.sqrt(diff_px**2 + diff_py**2)
    dx_norm = diff_x / np.sqrt(diff_x**2 + diff_py**2)
    dy_norm = diff_y / np.sqrt(diff_x**2 + diff_py**2)

    assert np.isclose(dpx_norm*dx_norm + dpy_norm*dy_norm, 0, atol=1e-6)

    # Match separation to 2 sigmas in IP2
    print(f'Knobs before matching: on_sep2 = {collider.vars["on_sep2"]._value}')
    collider.match(
        lines=['lhcb1', 'lhcb2'],
        ele_start=['e.ds.l2.b1', 's.ds.r2.b2'],
        ele_stop=['s.ds.r2.b1', 'e.ds.l2.b2'],
        twiss_init='preserve',
        targets=[
            xt.TargetSeparation(ip_name='ip2', separation_norm=3, plane='x', tol=1e-4,
                            nemitt_x=nemitt_x, nemitt_y=nemitt_y),
            # Preserve crossing angle
            xt.TargetList(['px', 'py'], at='ip2', line='lhcb1', value='preserve', tol=1e-7, scale=1e3),
            xt.TargetList(['px', 'py'], at='ip2', line='lhcb2', value='preserve', tol=1e-7, scale=1e3),
            # Close the bumps
            xt.TargetList(['x', 'y'], at='s.ds.r2.b1', line='lhcb1', value='preserve', tol=1e-5, scale=1),
            xt.TargetList(['px', 'py'], at='s.ds.r2.b1', line='lhcb1', value='preserve', tol=1e-5, scale=1e3),
            xt.TargetList(['x', 'y'], at='e.ds.l2.b2', line='lhcb2', value='preserve', tol=1e-5, scale=1),
            xt.TargetList(['px', 'py'], at='e.ds.l2.b2', line='lhcb2', value='preserve', tol=1e-5, scale=1e3),
        ],
        vary=
            [xt.Vary('on_sep2', step=1e-4),
            xt.VaryList([
                # correctors to control the crossing angles
                'corr_co_acbyvs4.l2b1', 'corr_co_acbyhs4.l2b1',
                'corr_co_acbyvs4.r2b2', 'corr_co_acbyhs4.r2b2',
                # correctors to close the bumps
                'corr_co_acbyvs4.l2b2', 'corr_co_acbyhs4.l2b2',
                'corr_co_acbyvs4.r2b1', 'corr_co_acbyhs4.r2b1',
                'corr_co_acbyhs5.l2b2', 'corr_co_acbyvs5.l2b2',
                'corr_co_acbchs5.r2b1', 'corr_co_acbcvs5.r2b1'],
                step=1e-7),
            ],
    )
    print(f'Knobs after matching: on_sep2 = {collider.vars["on_sep2"]._value}')

    tw_after_ip2_match = collider.twiss(lines=['lhcb1', 'lhcb2'])

    # Check normalized separation
    mean_betx = np.sqrt(tw_after_ip2_match['lhcb1']['betx', 'ip2']
                    *tw_after_ip2_match['lhcb2']['betx', 'ip2'])
    gamma0 = tw_after_ip2_match['lhcb1'].particle_on_co.gamma0[0]
    beta0 = tw_after_ip2_match['lhcb1'].particle_on_co.beta0[0]
    sigmax = np.sqrt(nemitt_x * mean_betx /gamma0 / beta0)

    assert np.isclose(collider.vars['on_sep2']._value/1000, 3*sigmax/2, rtol=1e-3, atol=0)


def test_tune_shift_single_6d_bb_lens_proton():

    num_particles = 1e11
    nemitt_x = 2.5e-6
    nemitt_y = 2.5e-6

    collider = xt.Multiline.from_json('collider_hllhc14_00.json')
    collider.build_trackers()

    # Switch off the crab cavities
    collider.vars['on_crab1'] = 0
    collider.vars['on_crab5'] = 0

    # Check that orbit is flat
    tw = collider.twiss(method='4d')
    assert np.max(np.abs(tw.lhcb1.x))< 1e-7

    # Install head-on only
    collider.discard_trackers()

    collider.install_beambeam_interactions(
        clockwise_line='lhcb1',
        anticlockwise_line='lhcb2',
        ip_names=['ip1'],
        num_long_range_encounters_per_side=[0],
        num_slices_head_on=1,
        harmonic_number=35640,
        bunch_spacing_buckets=10,
        sigmaz=1e-6
    )

    collider.build_trackers()
    # Switch on RF (assumes 6d)
    collider.vars['vrf400'] = 16

    collider.configure_beambeam_interactions(
        num_particles=num_particles,
        nemitt_x=nemitt_x, nemitt_y=nemitt_y,
        crab_strong_beam=False
    )

    collider.lhcb1.matrix_stability_tol = 1e-2
    collider.lhcb2.matrix_stability_tol = 1e-2

    tw_bb_on = collider.twiss(lines=['lhcb1', 'lhcb2'])
    collider.vars['beambeam_scale'] = 0
    tw_bb_off = collider.twiss(lines=['lhcb1', 'lhcb2'])

    tune_shift_x = tw_bb_on.lhcb1.qx - tw_bb_off.lhcb1.qx
    tune_shift_y = tw_bb_on.lhcb1.qy - tw_bb_off.lhcb1.qy

    # Analytical tune shift

    q0 = collider.lhcb1.particle_ref.q0
    mass0 = collider.lhcb1.particle_ref.mass0 # eV
    gamma0 = collider.lhcb1.particle_ref.gamma0[0]
    beta0 = collider.lhcb1.particle_ref.beta0[0]

    # classical particle radius
    r0 = 1 / (4 * np.pi * epsilon_0) * q0**2 * qe / mass0

    betx_weak = tw_bb_off.lhcb1['betx', 'ip1']
    bety_weak = tw_bb_off.lhcb1['bety', 'ip1']

    betx_strong = tw_bb_off.lhcb2['betx', 'ip1']
    bety_strong = tw_bb_off.lhcb2['bety', 'ip1']

    sigma_x_strong = np.sqrt(betx_strong * nemitt_x / beta0 / gamma0)
    sigma_y_strong = np.sqrt(bety_strong * nemitt_y / beta0 / gamma0)

    delta_qx = -(num_particles * r0 * betx_weak
                / (2 * np.pi * gamma0 * sigma_x_strong * (sigma_x_strong + sigma_y_strong)))
    delta_qy = -(num_particles * r0 * bety_weak
                / (2 * np.pi * gamma0 * sigma_y_strong * (sigma_x_strong + sigma_y_strong)))

    assert np.isclose(delta_qx, tune_shift_x, atol=0, rtol=1e-2)
    assert np.isclose(delta_qy, tune_shift_y, atol=0, rtol=1e-2)

def test_tune_shift_single_6d_bb_lens_ion():

    num_particles = 1.8e8
    nemitt_x = 1.65e-6
    nemitt_y = 1.65e-6

    collider = xt.Multiline.from_json('collider_hllhc14_00.json')
    collider.build_trackers()

    # Switch to ions
    for line in collider.lines.keys():
        collider[line].particle_ref = xp.Particles(mass0=193.6872729*1e9,
                                                q0=82, p0c=7e12*82) # Lead

    # Switch off the crab cavities
    collider.vars['on_crab1'] = 0
    collider.vars['on_crab5'] = 0

    # Check that orbit is flat
    tw = collider.twiss(method='4d')
    assert np.max(np.abs(tw.lhcb1.x))< 1e-7

    # Install head-on only
    collider.discard_trackers()

    collider.install_beambeam_interactions(
        clockwise_line='lhcb1',
        anticlockwise_line='lhcb2',
        ip_names=['ip1'],
        num_long_range_encounters_per_side=[0],
        num_slices_head_on=1,
        harmonic_number=35640,
        bunch_spacing_buckets=10,
        sigmaz=1e-6
    )

    collider.build_trackers()
    # Switch on RF (assumes 6d)
    collider.vars['vrf400'] = 16

    collider.configure_beambeam_interactions(
        num_particles=num_particles,
        nemitt_x=nemitt_x, nemitt_y=nemitt_y,
        crab_strong_beam=False
    )

    collider.lhcb1.matrix_stability_tol = 1e-2
    collider.lhcb2.matrix_stability_tol = 1e-2

    tw_bb_on = collider.twiss(lines=['lhcb1', 'lhcb2'])
    collider.vars['beambeam_scale'] = 0
    tw_bb_off = collider.twiss(lines=['lhcb1', 'lhcb2'])

    tune_shift_x = tw_bb_on.lhcb1.qx - tw_bb_off.lhcb1.qx
    tune_shift_y = tw_bb_on.lhcb1.qy - tw_bb_off.lhcb1.qy

    # Analytical tune shift

    q0 = collider.lhcb1.particle_ref.q0
    mass0 = collider.lhcb1.particle_ref.mass0 # eV
    gamma0 = collider.lhcb1.particle_ref.gamma0[0]
    beta0 = collider.lhcb1.particle_ref.beta0[0]

    # classical particle radius
    r0 = 1 / (4 * np.pi * epsilon_0) * q0**2 * qe / mass0

    betx_weak = tw_bb_off.lhcb1['betx', 'ip1']
    bety_weak = tw_bb_off.lhcb1['bety', 'ip1']

    betx_strong = tw_bb_off.lhcb2['betx', 'ip1']
    bety_strong = tw_bb_off.lhcb2['bety', 'ip1']

    sigma_x_strong = np.sqrt(betx_strong * nemitt_x / beta0 / gamma0)
    sigma_y_strong = np.sqrt(bety_strong * nemitt_y / beta0 / gamma0)

    delta_qx = -(num_particles * r0 * betx_weak
                / (2 * np.pi * gamma0 * sigma_x_strong * (sigma_x_strong + sigma_y_strong)))
    delta_qy = -(num_particles * r0 * bety_weak
                / (2 * np.pi * gamma0 * sigma_y_strong * (sigma_x_strong + sigma_y_strong)))

    assert np.isclose(delta_qx, tune_shift_x, atol=0, rtol=1e-2)
    assert np.isclose(delta_qy, tune_shift_y, atol=0, rtol=1e-2)

def test_tune_shift_single_4d_bb_lens_protons():

    num_particles = 1e11/3
    nemitt_x = 2.5e-6
    nemitt_y = 2.5e-6

    collider = xt.Multiline.from_json('collider_hllhc14_00.json')
    collider.build_trackers()

    # Switch off the crab cavities
    collider.vars['on_crab1'] = 0
    collider.vars['on_crab5'] = 0

    # Check that orbit is flat
    tw = collider.twiss(method='4d')
    assert np.max(np.abs(tw.lhcb1.x))< 1e-7

    # Install head-on only
    collider.discard_trackers()

    collider.install_beambeam_interactions(
        clockwise_line='lhcb1',
        anticlockwise_line='lhcb2',
        ip_names=['ip1'],
        num_long_range_encounters_per_side=[1],
        num_slices_head_on=1,
        harmonic_number=35640,
        bunch_spacing_buckets=1e-10, # To have them at the IP
        sigmaz=1e-6
    )

    collider.build_trackers()
    # Switch on RF (assumes 6d)
    collider.vars['vrf400'] = 16

    collider.configure_beambeam_interactions(
        num_particles=num_particles,
        nemitt_x=nemitt_x, nemitt_y=nemitt_y,
        crab_strong_beam=False
    )

    collider.lhcb1.matrix_stability_tol = 1e-2
    collider.lhcb2.matrix_stability_tol = 1e-2

    tw_bb_on = collider.twiss(lines=['lhcb1', 'lhcb2'])
    collider.vars['beambeam_scale'] = 0
    tw_bb_off = collider.twiss(lines=['lhcb1', 'lhcb2'])

    tune_shift_x = tw_bb_on.lhcb1.qx - tw_bb_off.lhcb1.qx
    tune_shift_y = tw_bb_on.lhcb1.qy - tw_bb_off.lhcb1.qy

    # Analytical tune shift

    q0 = collider.lhcb1.particle_ref.q0
    mass0 = collider.lhcb1.particle_ref.mass0 # eV
    gamma0 = collider.lhcb1.particle_ref.gamma0[0]
    beta0 = collider.lhcb1.particle_ref.beta0[0]

    # classical particle radius
    r0 = 1 / (4 * np.pi * epsilon_0) * q0**2 * qe / mass0

    betx_weak = tw_bb_off.lhcb1['betx', 'ip1']
    bety_weak = tw_bb_off.lhcb1['bety', 'ip1']

    betx_strong = tw_bb_off.lhcb2['betx', 'ip1']
    bety_strong = tw_bb_off.lhcb2['bety', 'ip1']

    sigma_x_strong = np.sqrt(betx_strong * nemitt_x / beta0 / gamma0)
    sigma_y_strong = np.sqrt(bety_strong * nemitt_y / beta0 / gamma0)

    delta_qx = -(num_particles * r0 * betx_weak
                / (2 * np.pi * gamma0 * sigma_x_strong * (sigma_x_strong + sigma_y_strong)))
    delta_qy = -(num_particles * r0 * bety_weak
                / (2 * np.pi * gamma0 * sigma_y_strong * (sigma_x_strong + sigma_y_strong)))

    assert np.isclose(delta_qx * 3, # head on + one long range per side
                    tune_shift_x, atol=0, rtol=1e-2)
    assert np.isclose(delta_qy * 3,  # head on + one long range per side
                    tune_shift_y, atol=0, rtol=1e-2)

def test_tune_shift_single_4d_bb_lens_ions():

    num_particles = 1.8e8
    nemitt_x = 1.65e-6
    nemitt_y = 1.65e-6

    collider = xt.Multiline.from_json('collider_hllhc14_00.json')
    collider.build_trackers()

    # Switch off the crab cavities
    collider.vars['on_crab1'] = 0
    collider.vars['on_crab5'] = 0

    # Switch to ions
    for line in collider.lines.keys():
        collider[line].particle_ref = xp.Particles(mass0=193.6872729*1e9,
                                                q0=82, p0c=7e12*82) # Lead

    # Check that orbit is flat
    tw = collider.twiss(method='4d')
    assert np.max(np.abs(tw.lhcb1.x))< 1e-7

    # Install head-on only
    collider.discard_trackers()

    collider.install_beambeam_interactions(
        clockwise_line='lhcb1',
        anticlockwise_line='lhcb2',
        ip_names=['ip1'],
        num_long_range_encounters_per_side=[1],
        num_slices_head_on=1,
        harmonic_number=35640,
        bunch_spacing_buckets=1e-10, # To have them at the IP
        sigmaz=1e-6
    )

    collider.build_trackers()
    # Switch on RF (assumes 6d)
    collider.vars['vrf400'] = 16

    collider.configure_beambeam_interactions(
        num_particles=num_particles,
        nemitt_x=nemitt_x, nemitt_y=nemitt_y,
        crab_strong_beam=False
    )

    collider.lhcb1.matrix_stability_tol = 1e-2
    collider.lhcb2.matrix_stability_tol = 1e-2

    tw_bb_on = collider.twiss(lines=['lhcb1', 'lhcb2'])
    collider.vars['beambeam_scale'] = 0
    tw_bb_off = collider.twiss(lines=['lhcb1', 'lhcb2'])

    tune_shift_x = tw_bb_on.lhcb1.qx - tw_bb_off.lhcb1.qx
    tune_shift_y = tw_bb_on.lhcb1.qy - tw_bb_off.lhcb1.qy

    # Analytical tune shift

    q0 = collider.lhcb1.particle_ref.q0
    mass0 = collider.lhcb1.particle_ref.mass0 # eV
    gamma0 = collider.lhcb1.particle_ref.gamma0[0]
    beta0 = collider.lhcb1.particle_ref.beta0[0]

    # classical particle radius
    r0 = 1 / (4 * np.pi * epsilon_0) * q0**2 * qe / mass0

    betx_weak = tw_bb_off.lhcb1['betx', 'ip1']
    bety_weak = tw_bb_off.lhcb1['bety', 'ip1']

    betx_strong = tw_bb_off.lhcb2['betx', 'ip1']
    bety_strong = tw_bb_off.lhcb2['bety', 'ip1']

    sigma_x_strong = np.sqrt(betx_strong * nemitt_x / beta0 / gamma0)
    sigma_y_strong = np.sqrt(bety_strong * nemitt_y / beta0 / gamma0)

    delta_qx = -(num_particles * r0 * betx_weak
                / (2 * np.pi * gamma0 * sigma_x_strong * (sigma_x_strong + sigma_y_strong)))
    delta_qy = -(num_particles * r0 * bety_weak
                / (2 * np.pi * gamma0 * sigma_y_strong * (sigma_x_strong + sigma_y_strong)))

    assert np.isclose(delta_qx * 3, # head on + one long range per side
                    tune_shift_x, atol=0, rtol=1e-2)
    assert np.isclose(delta_qy * 3,  # head on + one long range per side
                    tune_shift_y, atol=0, rtol=1e-2)


def test_apply_filling_scheme():

    collider = xt.Multiline.from_json('./collider_hllhc14_04.json')
    collider.build_trackers()

    filling_pattern_cw = np.zeros(3564, dtype=int)
    filling_pattern_acw = np.zeros(3564, dtype=int)

    # Some checks
    dframes = collider._bb_config['dataframes']
    assert (dframes['clockwise'].loc['bb_ho.c1b1_00', 'delay_in_slots'] == 0)
    assert (dframes['clockwise'].loc['bb_ho.c5b1_00', 'delay_in_slots'] == 0)
    assert (dframes['clockwise'].loc['bb_ho.c2b1_00', 'delay_in_slots'] == 891)
    assert (dframes['clockwise'].loc['bb_ho.c8b1_00', 'delay_in_slots'] == 2670)

    assert (dframes['anticlockwise'].loc['bb_ho.c1b2_00', 'delay_in_slots'] == 0)
    assert (dframes['anticlockwise'].loc['bb_ho.c5b2_00', 'delay_in_slots'] == 0)
    assert (dframes['anticlockwise'].loc['bb_ho.c2b2_00', 'delay_in_slots'] == 3564 - 891)
    assert (dframes['anticlockwise'].loc['bb_ho.c8b2_00', 'delay_in_slots'] == 3564 - 2670)

    assert (dframes['clockwise'].loc['bb_lr.r1b1_05', 'delay_in_slots'] == 0 + 5)
    assert (dframes['clockwise'].loc['bb_lr.r5b1_05', 'delay_in_slots'] == 0 + 5)
    assert (dframes['clockwise'].loc['bb_lr.r2b1_05', 'delay_in_slots'] == 891 + 5)
    assert (dframes['clockwise'].loc['bb_lr.r8b1_05', 'delay_in_slots'] == 2670 + 5)

    assert (dframes['anticlockwise'].loc['bb_lr.r1b2_05', 'delay_in_slots'] == 0 - 5)
    assert (dframes['anticlockwise'].loc['bb_lr.r5b2_05', 'delay_in_slots'] == 0 - 5)
    assert (dframes['anticlockwise'].loc['bb_lr.r2b2_05', 'delay_in_slots'] == 3564 - 891 - 5)
    assert (dframes['anticlockwise'].loc['bb_lr.r8b2_05', 'delay_in_slots'] == 3564 - 2670 - 5)

    assert (dframes['clockwise'].loc['bb_lr.l1b1_05', 'delay_in_slots'] == 0 - 5)
    assert (dframes['clockwise'].loc['bb_lr.l5b1_05', 'delay_in_slots'] == 0 - 5)
    assert (dframes['clockwise'].loc['bb_lr.l2b1_05', 'delay_in_slots'] == 891 - 5)
    assert (dframes['clockwise'].loc['bb_lr.l8b1_05', 'delay_in_slots'] == 2670 - 5)

    assert (dframes['anticlockwise'].loc['bb_lr.l1b2_05', 'delay_in_slots'] == 0 + 5)
    assert (dframes['anticlockwise'].loc['bb_lr.l5b2_05', 'delay_in_slots'] == 0 + 5)
    assert (dframes['anticlockwise'].loc['bb_lr.l2b2_05', 'delay_in_slots'] == 3564 - 891 + 5)
    assert (dframes['anticlockwise'].loc['bb_lr.l8b2_05', 'delay_in_slots'] == 3564 - 2670 + 5)

    twb1 = collider.lhcb1.twiss()
    twb2 = collider.lhcb2.twiss()

    ##############################################
    # Check with only one head-on in IP1 and IP5 #
    ##############################################

    filling_pattern_cw *= 0 # Reset
    filling_pattern_acw *= 0 # Reset
    filling_pattern_cw[1000] = 1
    filling_pattern_acw[1000] = 1

    i_bunch_cw = 1000
    i_bunch_acw = 1000

    collider.apply_filling_pattern(
        filling_pattern_cw=filling_pattern_cw,
        filling_pattern_acw=filling_pattern_acw,
        i_bunch_cw=i_bunch_cw, i_bunch_acw=i_bunch_acw)


    # Check that only head-on lenses in ip1 and ip5 are enabled
    all_bb_lenses_b1 = twb1.rows['bb_.*'].name
    assert np.sum([collider.lhcb1[nn].scale_strength for nn in all_bb_lenses_b1]) == 22 # 11 in IP1 and 11 in IP5
    all_bb_lenses_b2 = twb2.rows['bb_.*'].name
    assert np.sum([collider.lhcb2[nn].scale_strength for nn in all_bb_lenses_b2]) == 22 # 11 in IP1 and 11 in IP5

    assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*1b1.*'].name]) == 11 # 11 in IP1
    assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*5b1.*'].name]) == 11 # 11 in IP5
    assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*1b2.*'].name]) == 11 # 11 in IP1
    assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*5b2.*'].name]) == 11 # 11 in IP5
    assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*2b1.*'].name]) == 0 # 0 in IP2
    assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*8b1.*'].name]) == 0 # 0 in IP8
    assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*2b2.*'].name]) == 0 # 0 in IP2
    assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*8b2.*'].name]) == 0 # 0 in IP8

    assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_lr.*'].name]) == 0 # Long range
    assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_lr.*'].name]) == 0 # Long range


    ######################################
    # Check with only one head-on in IP8 #
    ######################################

    filling_pattern_cw *= 0 # Reset
    filling_pattern_acw *= 0 # Reset

    # These are supposed to collide in IP8 (checked with LPC tool)
    filling_pattern_cw[174] = 1
    filling_pattern_acw[2844] = 1

    i_bunch_cw = 174
    i_bunch_acw = 2844

    collider.apply_filling_pattern(
        filling_pattern_cw=filling_pattern_cw,
        filling_pattern_acw=filling_pattern_acw,
        i_bunch_cw=i_bunch_cw, i_bunch_acw=i_bunch_acw)

    all_bb_lenses_b1 = twb1.rows['bb_.*'].name
    assert np.sum([collider.lhcb1[nn].scale_strength for nn in all_bb_lenses_b1]) == 11 # 11 in IP8
    all_bb_lenses_b2 = twb2.rows['bb_.*'].name
    assert np.sum([collider.lhcb2[nn].scale_strength for nn in all_bb_lenses_b2]) == 11 # 11 in IP8

    assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*1b1.*'].name]) == 0 # IP1
    assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*5b1.*'].name]) == 0 # IP5
    assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*1b2.*'].name]) == 0 # IP1
    assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*5b2.*'].name]) == 0 # IP5
    assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*2b1.*'].name]) == 0 # IP2
    assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*8b1.*'].name]) == 11 # IP8
    assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*2b2.*'].name]) == 0  # IP2
    assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*8b2.*'].name]) == 11 # IP8

    assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_lr.*'].name]) == 0 # Long range
    assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_lr.*'].name]) == 0 # Long range


    ######################################
    # Check with only one head-on in IP2 #
    ######################################

    filling_pattern_cw *= 0 # Reset
    filling_pattern_acw *= 0 # Reset

    # These are supposed to collide in IP2 (checked with LPC tool)
    filling_pattern_cw[2952] = 1
    filling_pattern_acw[279] = 1

    i_bunch_cw = 2952
    i_bunch_acw = 279

    collider.apply_filling_pattern(
        filling_pattern_cw=filling_pattern_cw,
        filling_pattern_acw=filling_pattern_acw,
        i_bunch_cw=i_bunch_cw, i_bunch_acw=i_bunch_acw)

    all_bb_lenses_b1 = twb1.rows['bb_.*'].name
    assert np.sum([collider.lhcb1[nn].scale_strength for nn in all_bb_lenses_b1]) == 11 # 11 in IP2
    all_bb_lenses_b2 = twb2.rows['bb_.*'].name
    assert np.sum([collider.lhcb2[nn].scale_strength for nn in all_bb_lenses_b2]) == 11 # 11 in IP2

    assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*1b1.*'].name]) == 0 # IP1
    assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*5b1.*'].name]) == 0 # IP5
    assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*1b2.*'].name]) == 0 # IP1
    assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*5b2.*'].name]) == 0 # IP5
    assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*2b1.*'].name]) == 11 # IP2
    assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*8b1.*'].name]) == 0  # IP8
    assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*2b2.*'].name]) == 11 # IP2
    assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*8b2.*'].name]) == 0  # IP8

    assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_lr.*'].name]) == 0 # Long range
    assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_lr.*'].name]) == 0 # Long range

    ########################################################
    # Check with one long range on the left of IP1 and IP5 #
    ########################################################

    filling_pattern_cw *= 0 # Reset
    filling_pattern_acw *= 0 # Reset
    filling_pattern_cw[1000 + 5] = 1
    filling_pattern_acw[1000] = 1

    i_bunch_cw = 1000 + 5 # Long range expected on the left
    i_bunch_acw = 1000

    collider.apply_filling_pattern(
        filling_pattern_cw=filling_pattern_cw,
        filling_pattern_acw=filling_pattern_acw,
        i_bunch_cw=i_bunch_cw, i_bunch_acw=i_bunch_acw)


    # Check that only head-on lenses in ip1 and ip5 are enabled
    all_bb_lenses_b1 = twb1.rows['bb_.*'].name
    assert np.sum([collider.lhcb1[nn].scale_strength for nn in all_bb_lenses_b1]) == 2 # one long range in each of the main ips
    all_bb_lenses_b2 = twb2.rows['bb_.*'].name
    assert np.sum([collider.lhcb2[nn].scale_strength for nn in all_bb_lenses_b2]) == 2 # one long range in each of the main ips

    assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*1b1.*'].name]) == 0 # IP1
    assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*5b1.*'].name]) == 0 # IP5
    assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*1b2.*'].name]) == 0 # IP1
    assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*5b2.*'].name]) == 0 # IP5
    assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*2b1.*'].name]) == 0 # IP2
    assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*8b1.*'].name]) == 0 # IP8
    assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*2b2.*'].name]) == 0 # IP2
    assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*8b2.*'].name]) == 0 # IP8

    assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_lr.*'].name]) == 2 # Long range
    assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_lr.*'].name]) == 2 # Long range

    assert collider.lhcb1['bb_lr.l5b1_05'].scale_strength == 1
    assert collider.lhcb1['bb_lr.l1b1_05'].scale_strength == 1
    assert collider.lhcb2['bb_lr.l5b2_05'].scale_strength == 1
    assert collider.lhcb2['bb_lr.l1b2_05'].scale_strength == 1

    #################################################
    # Check with one long range on the right of IP2 #
    #################################################

    filling_pattern_cw *= 0 # Reset
    filling_pattern_acw *= 0 # Reset

    # These are supposed to collide in IP2 (checked with LPC tool)
    filling_pattern_cw[2952] = 1
    filling_pattern_acw[279 + 5] = 1

    i_bunch_cw = 2952
    i_bunch_acw = 279 + 5

    collider.apply_filling_pattern(
        filling_pattern_cw=filling_pattern_cw,
        filling_pattern_acw=filling_pattern_acw,
        i_bunch_cw=i_bunch_cw, i_bunch_acw=i_bunch_acw)

    all_bb_lenses_b1 = twb1.rows['bb_.*'].name
    assert np.sum([collider.lhcb1[nn].scale_strength for nn in all_bb_lenses_b1]) == 1 # IP2
    all_bb_lenses_b2 = twb2.rows['bb_.*'].name
    assert np.sum([collider.lhcb2[nn].scale_strength for nn in all_bb_lenses_b2]) == 1 # IP2

    assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*1b1.*'].name]) == 0 # IP1
    assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*5b1.*'].name]) == 0 # IP5
    assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*1b2.*'].name]) == 0 # IP1
    assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*5b2.*'].name]) == 0 # IP5
    assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*2b1.*'].name]) == 0 # IP2
    assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*8b1.*'].name]) == 0 # IP8
    assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*2b2.*'].name]) == 0 # IP2
    assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*8b2.*'].name]) == 0 # IP8

    assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_lr.*'].name]) == 1 # Long range
    assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_lr.*'].name]) == 1 # Long range

    assert collider.lhcb1['bb_lr.r2b1_05'].scale_strength == 1
    assert collider.lhcb2['bb_lr.r2b2_05'].scale_strength == 1

    #####################################
    # Many long ranges only on one side #
    #####################################

    filling_pattern_cw *= 0 # Reset
    filling_pattern_acw *= 0 # Reset

    filling_pattern_cw[1565 : 1565 + 48] = 1
    filling_pattern_acw[718 : 718 + 48] = 1
    filling_pattern_acw[1612 : 1612 + 48] = 1

    i_bunch_cw = 1612
    i_bunch_acw = 1612

    collider.apply_filling_pattern(
        filling_pattern_cw=filling_pattern_cw,
        filling_pattern_acw=filling_pattern_acw,
        i_bunch_cw=i_bunch_cw, i_bunch_acw=i_bunch_acw)

    all_bb_lenses_b1 = twb1.rows['bb_.*'].name
    assert np.sum([collider.lhcb1[nn].scale_strength for nn in all_bb_lenses_b1]) == (
        11 # head on IP1
        + 25 # long-range on one side of IP1
        + 11 # head on IP5
        + 25 # long-range on one side of IP5
        + 11 # head on IP2
        + 20 # long-range on one side of IP8
    )
    all_bb_lenses_b2 = twb2.rows['bb_.*'].name
    assert np.sum([collider.lhcb2[nn].scale_strength for nn in all_bb_lenses_b2]) == (
        11 # head on IP1
        + 25 # long-range on one side of IP1
        + 11 # head on IP5
        + 25 # long-range on one side of IP5
    )

    assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*1b1.*'].name]) == 11  # IP1
    assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*5b1.*'].name]) == 11  # IP5
    assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*1b2.*'].name]) == 11  # IP1
    assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*5b2.*'].name]) == 11  # IP5
    assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*2b1.*'].name]) == 0   # IP2
    assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_ho.*8b1.*'].name]) == 11  # IP8
    assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*2b2.*'].name]) == 0   # IP2
    assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_ho.*8b2.*'].name]) == 0   # IP8

    assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_lr.r1b1_.*'].name]) == 25
    assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_lr.l1b1_.*'].name]) == 0
    assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_lr.r1b2_.*'].name]) == 25
    assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_lr.l1b2_.*'].name]) == 0

    assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_lr.r5b1_.*'].name]) == 25
    assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_lr.l5b1_.*'].name]) == 0
    assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_lr.r5b2_.*'].name]) == 25
    assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_lr.l5b2_.*'].name]) == 0

    assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_lr.r2b1_.*'].name]) == 0
    assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_lr.l2b1_.*'].name]) == 0
    assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_lr.r2b2_.*'].name]) == 0
    assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_lr.l2b2_.*'].name]) == 0

    assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_lr.r8b1_.*'].name]) == 20
    assert np.sum([collider.lhcb1[nn].scale_strength for nn in twb1.rows['bb_lr.l8b1_.*'].name]) == 0
    assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_lr.r8b2_.*'].name]) == 0
    assert np.sum([collider.lhcb2[nn].scale_strength for nn in twb2.rows['bb_lr.l8b2_.*'].name]) == 0

    ###############################################
    # A case where all bb lenses should be active #
    ###############################################

    filling_pattern_cw *= 0 # Reset
    filling_pattern_acw *= 0 # Reset

    filling_pattern_cw[881 : 881 + 72] = 1
    filling_pattern_cw[1775 : 1775 + 72] = 1
    filling_pattern_cw[2669 : 2669 + 72] = 1

    filling_pattern_acw[881 : 881 + 72] = 1
    filling_pattern_acw[1775 : 1775 + 72] = 1
    filling_pattern_acw[2669 : 2669 + 72] = 1

    i_bunch_cw = 1775 + 36
    i_bunch_acw = 1775 + 36

    collider.apply_filling_pattern(
        filling_pattern_cw=filling_pattern_cw,
        filling_pattern_acw=filling_pattern_acw,
        i_bunch_cw=i_bunch_cw, i_bunch_acw=i_bunch_acw)

    all_bb_lenses_b1 = twb1.rows['bb_.*'].name
    assert np.all(np.array([collider.lhcb1[nn].scale_strength for nn in all_bb_lenses_b1]) == 1)
    all_bb_lenses_b2 = twb2.rows['bb_.*'].name
    assert np.all(np.array([collider.lhcb2[nn].scale_strength for nn in all_bb_lenses_b2]) == 1)

def test_multiline_match():
    collider = xt.Multiline.from_json('collider_hllhc14_02.json')
    collider.build_trackers()

    tw = collider.twiss(lines=['lhcb1', 'lhcb2'])
    assert tuple(tw._line_names) == ('lhcb1', 'lhcb2')
    assert 'mqs.23r2.b1' in tw.lhcb1.name
    assert 'mqs.23l4.b2' in tw.lhcb2.name
    assert tw.lhcb1['s', 'ip5'] < tw.lhcb1['s', 'ip6']
    assert tw.lhcb2['s', 'ip5'] > tw.lhcb2['s', 'ip6']
    assert np.isclose(tw.lhcb1.qx, 62.31, atol=1e-4, rtol=0)
    assert np.isclose(tw.lhcb1.qy, 60.32, atol=1e-4, rtol=0)
    assert np.isclose(tw.lhcb2.qx, 62.315, atol=1e-4, rtol=0)
    assert np.isclose(tw.lhcb2.qy, 60.325, atol=1e-4, rtol=0)

    tw_part = collider.twiss(
        lines=['lhcb1', 'lhcb2'],
        ele_start=['ip5', 'ip6'],
        ele_stop=['ip6', 'ip5'],
        twiss_init=[tw.lhcb1.get_twiss_init(at_element='ip5'), tw.lhcb2.get_twiss_init(at_element='ip6')]
    )

    # Add some asserts here

    collider.match(
        lines=['lhcb1', 'lhcb2'],
        vary=[
            xt.Vary('kqtf.b1', step=1e-8),
            xt.Vary('kqtd.b1', step=1e-8),
            xt.Vary('kqtf.b2', step=1e-8),
            xt.Vary('kqtd.b2', step=1e-8),
        ],
        targets = [
            xt.Target('qx', line='lhcb1', value=62.317, tol=1e-4),
            xt.Target('qy', line='lhcb1', value=60.327, tol=1e-4),
            xt.Target('qx', line='lhcb2', value=62.313, tol=1e-4),
            xt.Target('qy', line='lhcb2', value=60.323, tol=1e-4)
            ]
        )

    tw1 = collider.twiss(lines=['lhcb1', 'lhcb2'])
    assert tuple(tw1._line_names) == ('lhcb1', 'lhcb2')
    assert 'mqs.23r2.b1' in tw1.lhcb1.name
    assert 'mqs.23l4.b2' in tw1.lhcb2.name
    assert tw1.lhcb1['s', 'ip5'] < tw1.lhcb1['s', 'ip6']
    assert tw1.lhcb2['s', 'ip5'] > tw1.lhcb2['s', 'ip6']
    assert np.isclose(tw1.lhcb1.qx, 62.317, atol=1e-4, rtol=0)
    assert np.isclose(tw1.lhcb1.qy, 60.327, atol=1e-4, rtol=0)
    assert np.isclose(tw1.lhcb2.qx, 62.313, atol=1e-4, rtol=0)
    assert np.isclose(tw1.lhcb2.qy, 60.323, atol=1e-4, rtol=0)

    # Match bumps in the two likes
    collider.match(
        lines=['lhcb1', 'lhcb2'],
        ele_start=['mq.33l8.b1', 'mq.22l8.b2'],
        ele_stop=['mq.23l8.b1', 'mq.32l8.b2'],
        twiss_init='preserve',
        vary=[
            xt.VaryList([
                'acbv30.l8b1', 'acbv28.l8b1', 'acbv26.l8b1', 'acbv24.l8b1'],
                step=1e-10),
            xt.VaryList([
                'acbv29.l8b2', 'acbv27.l8b2', 'acbv25.l8b2', 'acbv23.l8b2'],
                step=1e-10),
        ],
        targets=[
            xt.Target('y', at='mb.b28l8.b1', line='lhcb1', value=3e-3, tol=1e-4, scale=1),
            xt.Target('py', at='mb.b28l8.b1', line='lhcb1', value=0, tol=1e-6, scale=1000),
            xt.Target('y', at='mb.b27l8.b2', line='lhcb2', value=2e-3, tol=1e-4, scale=1),
            xt.Target('py', at='mb.b27l8.b2', line='lhcb2', value=0, tol=1e-6, scale=1000),
            # I want the bump to be closed
            xt.TargetList(['y'], at='mq.23l8.b1', line='lhcb1', value='preserve', tol=1e-6, scale=1),
            xt.TargetList(['py'], at='mq.23l8.b1', line='lhcb1', value='preserve', tol=1e-7, scale=1000),
            xt.TargetList(['y'], at='mq.32l8.b2', line='lhcb2', value='preserve', tol=1e-6, scale=1),
            xt.Target('py', at='mq.32l8.b2', line='lhcb2', value='preserve', tol=1e-10, scale=1000),
        ]
    )
    tw_bump = collider.twiss(lines=['lhcb1', 'lhcb2'])

    tw_before = tw1.lhcb1
    assert np.isclose(tw_bump.lhcb1['y', 'mb.b28l8.b1'], 3e-3, atol=1e-4)
    assert np.isclose(tw_bump.lhcb1['py', 'mb.b28l8.b1'], 0, atol=1e-6)
    assert np.isclose(tw_bump.lhcb1['y', 'mq.23l8.b1'], tw_before['y', 'mq.23l8.b1'], atol=1e-6)
    assert np.isclose(tw_bump.lhcb1['py', 'mq.23l8.b1'], tw_before['py', 'mq.23l8.b1'], atol=1e-7)
    assert np.isclose(tw_bump.lhcb1['y', 'mq.33l8.b1'], tw_before['y', 'mq.33l8.b1'], atol=1e-6)
    assert np.isclose(tw_bump.lhcb1['py', 'mq.33l8.b1'], tw_before['py', 'mq.33l8.b1'], atol=1e-7)

    tw_before = tw1.lhcb2
    assert np.isclose(tw_bump.lhcb2['y', 'mb.b27l8.b2'], 2e-3, atol=1e-4)
    assert np.isclose(tw_bump.lhcb2['py', 'mb.b27l8.b2'], 0, atol=1e-6)
    assert np.isclose(tw_bump.lhcb2['y', 'mq.32l8.b2'], tw_before['y', 'mq.33l8.b2'], atol=1e-6)
    assert np.isclose(tw_bump.lhcb2['py', 'mq.32l8.b2'], tw_before['py', 'mq.33l8.b2'], atol=1e-7)
    assert np.isclose(tw_bump.lhcb2['y', 'mq.22l8.b2'], tw_before['y', 'mq.23l8.b2'], atol=1e-6)
    assert np.isclose(tw_bump.lhcb2['py', 'mq.22l8.b2'], tw_before['py', 'mq.23l8.b2'], atol=1e-7)
