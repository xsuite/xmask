import numpy as np

from cpymad.madx import Madx
import xtrack as xt

import xmask as xm
import xmask.lhc as xmlhc
import yaml

# Import user-defined optics-specific tools
from _complementary_run3_ions import (
    _config_ion_yaml_str, build_sequence, apply_optics, orbit_correction_config,
    check_optics_orbit_etc)

def test_lhc_ion_0_create_collider():
    # Read config file
    config = yaml.safe_load(_config_ion_yaml_str)
    config_mad_model = config['config_mad']

    # Make mad environment
    xm.make_mad_environment(links=config_mad_model['links'])

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
    sigmaz=0.076)

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

    assert np.isclose(tw1_b1.dqx, tw0_b1.dqx, atol=1e-4, rtol=0)
    assert np.isclose(tw1_b1.dqy, tw0_b1.dqy, atol=1e-4, rtol=0)
    assert np.isclose(tw1_b2.dqx, tw0_b2.dqx, atol=1e-4, rtol=0)
    assert np.isclose(tw1_b2.dqy, tw0_b2.dqy, atol=1e-4, rtol=0)

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