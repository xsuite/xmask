import yaml
from pathlib import Path
import numpy as np
from cpymad.madx import Madx

import xtrack as xt

import pymaskmx as pm
import pymaskmx.lhc as pmlhc

# We assume that the tests will be run in order. In case of issues we could use
# https://pypi.org/project/pytest-order/ to enforce the order.

test_data_dir = Path(__file__).parent.parent / "test_data"

def test_hllhc14_0_create_collider():
    # Make mad environment
    pm.make_mad_environment(links={
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
    collider = pmlhc.build_xsuite_collider(
        sequence_b1=mad_b1b2.sequence.lhcb1,
        sequence_b2=mad_b1b2.sequence.lhcb2,
        sequence_b4=mad_b4.sequence.lhcb2,
        beam_config={'lhcb1':{'beam_energy_tot': 7000},
                     'lhcb2':{'beam_energy_tot': 7000}},
        enable_imperfections=False,
        enable_knob_synthesis='_mock_for_testing',
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
    num_long_range_encounters_per_side=[25, 20, 25, 20],
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

    assert np.isclose(tw1_b1.dqx, tw0_b1.dqx, atol=1e-4, rtol=0)
    assert np.isclose(tw1_b1.dqy, tw0_b1.dqy, atol=1e-4, rtol=0)
    assert np.isclose(tw1_b2.dqx, tw0_b2.dqx, atol=1e-4, rtol=0)
    assert np.isclose(tw1_b2.dqy, tw0_b2.dqy, atol=1e-4, rtol=0)

    for ipn in [1, 2, 3, 4, 5, 6, 7, 8]:
        assert np.isclose(tw1_b1[f'ip{ipn}', 'betx'], tw0_b1[f'ip{ipn}', 'betx'], rtol=1e-5, atol=0)
        assert np.isclose(tw1_b1[f'ip{ipn}', 'bety'], tw0_b1[f'ip{ipn}', 'bety'], rtol=1e-5, atol=0)
        assert np.isclose(tw1_b2[f'ip{ipn}', 'betx'], tw0_b2[f'ip{ipn}', 'betx'], rtol=1e-5, atol=0)
        assert np.isclose(tw1_b2[f'ip{ipn}', 'bety'], tw0_b2[f'ip{ipn}', 'bety'], rtol=1e-5, atol=0)

        assert np.isclose(tw1_b1[f'ip{ipn}', 'px'], tw0_b1[f'ip{ipn}', 'px'], rtol=1e-9, atol=0)
        assert np.isclose(tw1_b1[f'ip{ipn}', 'py'], tw0_b1[f'ip{ipn}', 'py'], rtol=1e-9, atol=0)
        assert np.isclose(tw1_b2[f'ip{ipn}', 'px'], tw0_b2[f'ip{ipn}', 'px'], rtol=1e-9, atol=0)
        assert np.isclose(tw1_b2[f'ip{ipn}', 'py'], tw0_b2[f'ip{ipn}', 'py'], rtol=1e-9, atol=0)

        assert np.isclose(tw1_b1[f'ip{ipn}', 's'], tw0_b1[f'ip{ipn}', 's'], rtol=1e-10, atol=0)
        assert np.isclose(tw1_b2[f'ip{ipn}', 's'], tw0_b2[f'ip{ipn}', 's'], rtol=1e-10, atol=0)


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

        pm.machine_tuning(line=collider[line_name],
            enable_closed_orbit_correction=True,
            enable_linear_coupling_correction=True,
            enable_tune_correction=True,
            enable_chromaticity_correction=True,
            knob_names=knob_names,
            targets=targets,
            line_co_ref=collider[line_name+'_co_ref'],
            co_corr_config=orbit_correction_config[line_name])

    collider.to_json('collider_hllhc14_02.json')

    for line_name in ['lhcb1', 'lhcb2']:

        assert collider[line_name].particle_ref.q0 == 1
        assert np.isclose(collider[line_name].particle_ref.p0c, 7e12,
                        atol=0, rtol=1e-5)
        assert np.isclose(collider[line_name].particle_ref.mass0, 0.9382720813e9,
                            atol=0, rtol=1e-5)

        tw = collider[line_name].twiss()

        if line_name == 'lhcb1':
            assert np.isclose(tw.qx, 62.31, atol=1e-4, rtol=0)
            assert np.isclose(tw.qy, 60.32, atol=1e-4, rtol=0)
            assert np.isclose(tw.dqx, 5, atol=0.1, rtol=0)
            assert np.isclose(tw.dqy, 7, atol=0.1, rtol=0)
        elif line_name == 'lhcb2':
            assert np.isclose(tw.qx, 62.315, atol=1e-4, rtol=0)
            assert np.isclose(tw.qy, 60.325, atol=1e-4, rtol=0)
            assert np.isclose(tw.dqx, 6, atol=0.1, rtol=0)
            assert np.isclose(tw.dqy, 8, atol=0.1, rtol=0)
        else:
            raise ValueError(f'Unknown line name {line_name}')

        assert np.isclose(tw.qs, 0.00212, atol=1e-4, rtol=0) # Checks that RF is well set

        assert np.isclose(tw.c_minus, 0, atol=1e-4, rtol=0)
        assert np.allclose(tw.zeta, 0, rtol=0, atol=1e-4) # Check RF phase

        # Check separations
        assert np.isclose(tw['ip1', 'x'], 0, rtol=0, atol=5e-8) # sigma is 4e-6
        assert np.isclose(tw['ip1', 'y'], 0, rtol=0, atol=5e-8) # sigma is 4e-6
        assert np.isclose(tw['ip5', 'x'], 0, rtol=0, atol=5e-8) # sigma is 4e-6
        assert np.isclose(tw['ip5', 'y'], 0, rtol=0, atol=5e-8) # sigma is 4e-6

        assert np.isclose(tw['ip2', 'x'],
                -0.138e-3 * {'lhcb1': 1, 'lhcb2': 1}[line_name], # set separation
                rtol=0, atol=4e-6)
        assert np.isclose(tw['ip2', 'y'], 0, rtol=0, atol=5e-8)

        assert np.isclose(tw['ip8', 'x'], 0, rtol=0, atol=5e-8)
        assert np.isclose(tw['ip8', 'y'],
                -0.043e-3 * {'lhcb1': 1, 'lhcb2': -1}[line_name], # set separation
                rtol=0, atol=5e-8)

        # Check crossing angles
        assert np.isclose(tw['ip1', 'px'],
                250e-6 * {'lhcb1': 1, 'lhcb2': -1}[line_name], rtol=0, atol=0.5e-6)
        assert np.isclose(tw['ip1', 'py'], 0, rtol=0, atol=0.5e-6)
        assert np.isclose(tw['ip5', 'px'], 0, rtol=0, atol=0.5e-6)
        assert np.isclose(tw['ip5', 'py'], 250e-6, rtol=0, atol=0.5e-6)

        assert np.isclose(tw['ip2', 'px'], 0, rtol=0, atol=0.5e-6)
        assert np.isclose(tw['ip2', 'py'], -100e-6 , rtol=0, atol=0.5e-6) # accounts for spectrometer

        assert np.isclose(tw['ip8', 'px'],
                -115e-6* {'lhcb1': 1, 'lhcb2': -1}[line_name], rtol=0, atol=0.5e-6) # accounts for spectrometer
        assert np.isclose(tw['ip8', 'py'], 2e-6, rtol=0, atol=0.5e-6) # small effect from spectrometer (titled)

        assert np.isclose(tw['ip1', 'betx'], 15e-2, rtol=2e-2, atol=0) # beta beating coming from on_disp
        assert np.isclose(tw['ip1', 'bety'], 15e-2, rtol=3e-2, atol=0)
        assert np.isclose(tw['ip5', 'betx'], 15e-2, rtol=2e-2, atol=0)
        assert np.isclose(tw['ip5', 'bety'], 15e-2, rtol=2e-2, atol=0)

        assert np.isclose(tw['ip2', 'betx'], 10., rtol=4e-2, atol=0)
        assert np.isclose(tw['ip2', 'bety'], 10., rtol=3e-2, atol=0)

        assert np.isclose(tw['ip8', 'betx'], 1.5, rtol=3e-2, atol=0)
        assert np.isclose(tw['ip8', 'bety'], 1.5, rtol=2e-2, atol=0)

        # Check crab cavities
        z_crab_test = 1e-2
        phi_crab_1 = ((
            collider[line_name].twiss(method='4d', zeta0=z_crab_test)['ip1', 'x']
        - collider[line_name].twiss(method='4d', zeta0=-z_crab_test)['ip1', 'x'])
        / 2 / z_crab_test)

        phi_crab_5 = ((
            collider[line_name].twiss(method='4d', zeta0=z_crab_test)['ip5', 'y']
        - collider[line_name].twiss(method='4d', zeta0=-z_crab_test)['ip5', 'y'])
        / 2 / z_crab_test)

        assert np.isclose(phi_crab_1, -190e-6 * {'lhcb1': 1, 'lhcb2': -1}[line_name],
                        rtol=1e-2, atol=0)
        assert np.isclose(phi_crab_5, -170e-6, rtol=1e-2, atol=0)

        # Check one octupole strength
        assert np.isclose(collider['lhcb1']['mo.33l4.b1'].knl[3], -2.2169*200/235,
                          rtol=1e-3, atol=0)
        assert np.isclose(collider['lhcb2']['mo.33r4.b2'].knl[3], -2.2169,
                          rtol=1e-3, atol=0)

def build_sequence(mad, mylhcbeam, **kwargs):

    # Select beam
    mad.input(f'mylhcbeam = {mylhcbeam}')

    mad.input(

    f'''
    ! Get the toolkit
    call,file=
        "acc-models-lhc/toolkit/macro.madx";
    '''
    '''
    ! Build sequence
    option, -echo,-warn,-info;
    if (mylhcbeam==4){
        call,file="acc-models-lhc/../runIII/lhcb4.seq";
    } else {
        call,file="acc-models-lhc/../runIII/lhc.seq";
    };
    option, -echo, warn,-info;
    '''
    f'''
    !Install HL-LHC
    call, file=
        "acc-models-lhc/hllhc_sequence.madx";
    '''
    '''
    ! Slice nominal sequence
    exec, myslice;
    ''')

    pmlhc.install_errors_placeholders_hllhc(mad)

    mad.input(
    '''
    !Cycling w.r.t. to IP3 (mandatory to find closed orbit in collision in the presence of errors)
    if (mylhcbeam<3){
        seqedit, sequence=lhcb1; flatten; cycle, start=IP3; flatten; endedit;
    };
    seqedit, sequence=lhcb2; flatten; cycle, start=IP3; flatten; endedit;

    ! Install crab cavities (they are off)
    call, file='acc-models-lhc/toolkit/enable_crabcavities.madx';
    on_crab1 = 0;
    on_crab5 = 0;

    ! Set twiss formats for MAD-X parts (macro from opt. toolkit)
    exec, twiss_opt;

    '''
    )


def apply_optics(mad, optics_file):
    mad.call(optics_file)
    # A knob redefinition
    mad.input('on_alice := on_alice_normalized * 7000./nrj;')
    mad.input('on_lhcb := on_lhcb_normalized * 7000./nrj;')

orbit_correction_config = {}
orbit_correction_config['lhcb1'] = {
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

orbit_correction_config['lhcb2'] = {
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

knob_settings_yaml_str = """
  # Orbit knobs
  on_x1: 250            # [urad]
  on_sep1: 0            # [mm]
  on_x2: -170           # [urad]
  on_sep2: 0.138        # [mm]
  on_x5: 250            # [urad]
  on_sep5: 0            # [mm]
  on_x8: -250           # [urad]
  on_sep8: -0.043       # [mm]
  on_a1: 0              # [urad]
  on_o1: 0              # [mm]
  on_a2: 0              # [urad]
  on_o2: 0              # [mm]
  on_a5: 0              # [urad]
  on_o5: 0              # [mm]
  on_a8: 0              # [urad]
  on_o8: 0              # [mm]
  on_disp: 1            # Value to choose could be optics-dependent

  # Crab cavities
  on_crab1: -190        # [urad]
  on_crab5: -170        # [urad]

  # Magnets of the experiments
  on_alice_normalized: 1
  on_lhcb_normalized: 1
  on_sol_atlas: 0
  on_sol_cms: 0
  on_sol_alice: 0

  # RF voltage and phases
  vrf400:       16.0            # [MV]
  lagrf400.b1:   0.5            # [rad]
  lagrf400.b2:   0.             # [rad]

  # Octupoles
  i_oct_b1:     -200            # [A]
  i_oct_b2:     -235            # [A]
"""

knob_names_yaml_str = """
lhcb1:
    q_knob_1: kqtf.b1
    q_knob_2: kqtd.b1
    dq_knob_1: ksf.b1
    dq_knob_2: ksd.b1
    c_minus_knob_1: c_minus_re_b1
    c_minus_knob_2: c_minus_im_b1

lhcb2:
    q_knob_1: kqtf.b2
    q_knob_2: kqtd.b2
    dq_knob_1: ksf.b2
    dq_knob_2: ksd.b2
    c_minus_knob_1: c_minus_re_b2
    c_minus_knob_2: c_minus_im_b2
"""
tune_chroma_yaml_str = """
qx:
  lhcb1: 62.31
  lhcb2: 62.315
qy:
  lhcb1: 60.32
  lhcb2: 60.325
dqx:
  lhcb1: 5
  lhcb2: 6
dqy:
  lhcb1: 7
  lhcb2: 8

"""