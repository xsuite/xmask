import numpy as np

import xobjects as xo

_config_ion_yaml_str = """
config_mad:
    # Links to be made for tools and scripts
    links:
      optics_runII: ../test_data/lhc_ion/runII
      optics_runIII: ../test_data/lhc_ion/runIII

    # Optics file
    optics_file: optics_runII/2018/ION/opticsfile.21

    # Beam parameters
    beam_config:
      lhcb1:
        # Lead ions
        beam_energy_tot: 574000           # [GeV] corresponds to 7 Z TeV
        particle_mass: 193.6872729        # [GeV]
        particle_charge: 82
      lhcb2:
        # Lead ions
        beam_energy_tot: 574000           # [GeV] corresponds to 7 Z TeV
        particle_mass: 193.6872729        # [GeV]
        particle_charge: 82

    # Enable machine imperfections
    enable_imperfections: false

    # Enable knob synthesis (for coupling correction, if no imperfections)
    enable_knob_synthesis: false

    # Optics version, for choice of correction algorithms
    # (ver_lhc_run or ver_hllhc_optics)
    ver_hllhc_optics: 0.0
    ver_lhc_run: null

    # Parameters for machine imperfections
    pars_for_imperfections:
      par_myseed: 1
      par_correct_for_D2: 0
      par_correct_for_MCBX: 0
      par_on_errors_LHC: 1
      par_off_errors_Q4_inIP15: 0
      par_off_errors_Q5_inIP15: 0
      par_on_errors_MBH: 1
      par_on_errors_Q4: 1
      par_on_errors_D2: 1
      par_on_errors_D1: 1
      par_on_errors_IT: 1
      par_on_errors_MCBRD: 0
      par_on_errors_MCBXF: 0
      par_on_errors_NLC: 0
      par_write_errortable: 1

config_knobs_and_tuning:

    knob_settings:

      # Orbit knobs
      on_x1: 170          # [urad]
      on_sep1: 1.0e-3     # [mm]
      on_x2: 170          # [urad]
      on_sep2: 1.0e-3     # [mm]
      on_x5: 170          # [urad]
      on_sep5: 1.0e-3     # [mm]
      on_x8: 170          # [urad]
      on_sep8: 1.0e-03    # [mm]
      on_ov2: 0.0         # [mm]
      on_ov5: 0.0         # [mm]
      on_disp: 0
      on_alice_normalized: -1
      on_lhcb_normalized: -1
      on_sol_atlas: 0
      on_sol_cms: 0
      on_sol_alice: 0

      # RF voltage and phases
      vrf400:       1148            # [MV] (14 * 82 = 1148 MV)
      lagrf400.b1:   0.5            # [rad]
      lagrf400.b2:   0.             # [rad]

      # Octupoles
      i_oct_b1:     250             # [A]
      i_oct_b2:     250             # [A]

    closed_orbit_correction:
      lhcb1: corr_co_lhcb1.json
      lhcb2: corr_co_lhcb2.json

    # Tunes and chromaticities
    qx:
      lhcb1: 62.31
      lhcb2: 62.31
    qy:
      lhcb1: 60.32
      lhcb2: 60.32
    dqx:
      lhcb1: 10
      lhcb2: 10
    dqy:
      lhcb1: 10
      lhcb2: 10

    knob_names:
      lhcb1:
        q_knob_1: dqx.b1_sq
        q_knob_2: dqy.b1_sq
        dq_knob_1: dqpx.b1_sq
        dq_knob_2: dqpy.b1_sq
        c_minus_knob_1: cmrs.b1_sq
        c_minus_knob_2: cmis.b1_sq
      lhcb2:
        q_knob_1: dqx.b2_sq
        q_knob_2: dqy.b2_sq
        dq_knob_1: dqpx.b2_sq
        dq_knob_2: dqpy.b2_sq
        c_minus_knob_1: cmrs.b2_sq
        c_minus_knob_2: cmis.b2_sq

config_beambeam:
    bunch_spacing_buckets: 10
    num_slices_head_on: 11
    num_long_range_encounters_per_side:
      ip1: 25
      ip2: 20
      ip5: 25
      ip8: 20
    sigma_z: 0.0824
    num_particles_per_bunch: 180000000.0
    nemitt_x: 1.65e-6
    nemitt_y: 1.65e-6
    mask_with_filling_pattern:
      pattern_fname: filling.json
      i_bunch_b1: 488
      i_bunch_b2: 488

config_lumi_leveling:

    ip1:
      luminosity: 6.4e+27
      num_colliding_bunches: 1088
      impose_separation_orthogonal_to_crossing: false
      knobs:
        - on_sep1
      bump_range:
        lhcb1:
          - e.ds.l1.b1
          - s.ds.r1.b1
        lhcb2:
          - s.ds.r1.b2
          - e.ds.l1.b2
      preserve_angles_at_ip: true
      preserve_bump_closure: true
      corrector_knob_names:
        - corr_co_acbyvs4.l1b1
        - corr_co_acbyhs4.l1b1
        - corr_co_acbyvs4.r1b2
        - corr_co_acbyhs4.r1b2
        - corr_co_acbyhs4.r1b1
        - corr_co_acbyvs4.r1b1
        - corr_co_acbyhs4.l1b2
        - corr_co_acbyvs4.l1b2
        - corr_co_acbcv5.l1b1
        - corr_co_acbch5.r1b1
        - corr_co_acbcv5.r1b2
        - corr_co_acbch5.l1b2

    ip5:
      luminosity: 6.4e+27
      num_colliding_bunches: 1088
      impose_separation_orthogonal_to_crossing: false
      impose_separation_orthogonal_to_crossing: false
      knobs:
        - on_sep5
      bump_range:
        lhcb1:
          - e.ds.l5.b1
          - s.ds.r5.b1
        lhcb2:
          - s.ds.r5.b2
          - e.ds.l5.b2
      preserve_angles_at_ip: true
      preserve_bump_closure: true
      corrector_knob_names:
        - corr_co_acbcv5.l5b1
        - corr_co_acbch5.r5b1
        - corr_co_acbyhs4.l5b1
        - corr_co_acbyhs4.r5b1
        - corr_co_acbyvs4.l5b1
        - corr_co_acbyvs4.r5b1
        - corr_co_acbcv5.r5b2
        - corr_co_acbch5.l5b2
        - corr_co_acbyhs4.l5b2
        - corr_co_acbyhs4.r5b2
        - corr_co_acbyvs4.l5b2
        - corr_co_acbyvs4.r5b2

    ip2:
      luminosity: 6.4e+27
      num_colliding_bunches: 1088
      impose_separation_orthogonal_to_crossing: false
      impose_separation_orthogonal_to_crossing: false
      knobs:
        - on_sep2
      bump_range:
        lhcb1:
          - e.ds.l2.b1
          - s.ds.r2.b1
        lhcb2:
          - s.ds.r2.b2
          - e.ds.l2.b2
      preserve_angles_at_ip: true
      preserve_bump_closure: true
      corrector_knob_names:
       # to preserve angles at ip
       - corr_co_acbyvs4.l2b1
       - corr_co_acbyhs4.l2b1
       - corr_co_acbyvs4.r2b2
       - corr_co_acbyhs4.r2b2
       # to close the bumps
       - corr_co_acbyvs4.l2b2
       - corr_co_acbyhs4.l2b2
       - corr_co_acbyvs4.r2b1
       - corr_co_acbyhs4.r2b1
       - corr_co_acbyhs5.l2b2
       - corr_co_acbyvs5.l2b2
       - corr_co_acbchs5.r2b1
       - corr_co_acbcvs5.r2b1

    ip8:
      luminosity: 1.0e+27
      num_colliding_bunches: 398
      impose_separation_orthogonal_to_crossing: false
      knobs:
        - on_sep8
      bump_range:
        lhcb1:
          - e.ds.l8.b1
          - s.ds.r8.b1
        lhcb2:
          - s.ds.r8.b2
          - e.ds.l8.b2
      preserve_angles_at_ip: true
      preserve_bump_closure: true
      corrector_knob_names:
       # to preserve angles at ip
       - corr_co_acbyvs4.l8b1
       - corr_co_acbyhs4.l8b1
       - corr_co_acbyvs4.r8b2
       - corr_co_acbyhs4.r8b2
       # to close the bumps
       - corr_co_acbyvs4.l8b2
       - corr_co_acbyhs4.l8b2
       - corr_co_acbyvs4.r8b1
       - corr_co_acbyhs4.r8b1
       - corr_co_acbcvs5.l8b2
       - corr_co_acbchs5.l8b2
       - corr_co_acbyvs5.r8b1
       - corr_co_acbyhs5.r8b1

"""

import warnings

def build_sequence(mad, mylhcbeam, **kwargs):

    # Select beam

    #slicefactor = 2 # For testing
    slicefactor = 8 # For production

    mylhcbeam = int(mylhcbeam)

    mad.input('ver_lhc_run = 3')

    mad.input(f'mylhcbeam = {mylhcbeam}')
    mad.input('option, -echo,warn, -info;')

    # optics dependent macros (for splitting)
    mad.call('optics_runII/2018/toolkit/macro.madx')

    # # Redefine macros
    # _redefine_crossing_save_disable_restore(mad)

    # # optics independent macros
    # mad.call('tools/optics_indep_macros.madx')

    assert mylhcbeam in [1, 2, 4], "Invalid mylhcbeam (it should be in [1, 2, 4])"

    if mylhcbeam in [1, 2]:
        mad.call('optics_runII/2018/lhc_as-built.seq')
    else:
        mad.call('optics_runII/2018/lhcb4_as-built.seq')

    # New IR7 MQW layout and cabling
    mad.call('optics_runIII/RunIII_dev/IR7-Run3seqedit.madx')

    # Makethin part
    if slicefactor > 0:
        # the variable in the macro is slicefactor
        mad.input(f'slicefactor={slicefactor};')
        mad.call('optics_runII/2018/toolkit/myslice.madx')
        mad.beam()
        for my_sequence in ['lhcb1','lhcb2']:
            if my_sequence in list(mad.sequence):
                mad.input(f'use, sequence={my_sequence}; makethin,'
                     f'sequence={my_sequence}, style=teapot, makedipedge=true;')
    else:
        warnings.warn('The sequences are not thin!')

    # Cycling w.r.t. to IP3 (mandatory to find closed orbit in collision in the presence of errors)
    for my_sequence in ['lhcb1','lhcb2']:
        if my_sequence in list(mad.sequence):
            mad.input(f'seqedit, sequence={my_sequence}; flatten;'
                        'cycle, start=IP3; flatten; endedit;')

def apply_optics(mad, optics_file):
    mad.call(optics_file)
    mad.call('optics_runIII/ir7_strengths.madx')
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

def check_optics_orbit_etc(collider, line_names, sep_h_ip1, sep_h_ip2,
                           sep_v_ip5, sep_v_ip8):

    for line_name in line_names:

        assert collider[line_name].particle_ref.q0 == 82
        xo.assert_allclose(collider[line_name].particle_ref.p0c, 5.73999967e+14,
                        atol=0, rtol=1e-5)
        xo.assert_allclose(collider[line_name].particle_ref.mass0, 193687272900.0,
                            atol=0, rtol=1e-5)

        tw = collider[line_name].twiss()

        if line_name == 'lhcb1':
            xo.assert_allclose(tw.qx, 62.31, atol=1e-4, rtol=0)
            xo.assert_allclose(tw.qy, 60.32, atol=1e-4, rtol=0)
            xo.assert_allclose(tw.dqx, 10, atol=0.1, rtol=0)
            xo.assert_allclose(tw.dqy, 10, atol=0.1, rtol=0)
        elif line_name == 'lhcb2':
            xo.assert_allclose(tw.qx, 62.31, atol=1e-4, rtol=0)
            xo.assert_allclose(tw.qy, 60.32, atol=1e-4, rtol=0)
            xo.assert_allclose(tw.dqx, 10, atol=0.1, rtol=0)
            xo.assert_allclose(tw.dqy, 10, atol=0.1, rtol=0)
        else:
            raise ValueError(f'Unknown line name {line_name}')

        xo.assert_allclose(tw.qs, 0.00198862, atol=1e-4, rtol=0) # Checks that RF is well set

        xo.assert_allclose(tw.c_minus, 0, atol=1e-4, rtol=0)
        assert np.allclose(tw.zeta, 0, rtol=0, atol=1e-4) # Check RF phase

        if sep_h_ip1 is not None:
            xo.assert_allclose(tw['x', 'ip1'],
                    #-0.138e-3 * {'lhcb1': 1, 'lhcb2': 1}[line_name], # set separation
                    sep_h_ip1 * {'lhcb1': 1, 'lhcb2': 1}[line_name],
                    rtol=0, atol=1e-7)
        xo.assert_allclose(tw['y', 'ip1'], 0, rtol=0, atol=1e-7)

        if sep_h_ip2 is not None:
            xo.assert_allclose(tw['x', 'ip2'],
                    #-0.138e-3 * {'lhcb1': 1, 'lhcb2': 1}[line_name], # set separation
                    sep_h_ip2 * {'lhcb1': 1, 'lhcb2': 1}[line_name],
                    rtol=0, atol=1e-7)
        xo.assert_allclose(tw['y', 'ip2'], 0, rtol=0, atol=1e-7)

        xo.assert_allclose(tw['x', 'ip8'], 0, rtol=0, atol=1e-7)
        if sep_v_ip8 is not None:
            xo.assert_allclose(tw['y', 'ip8'],
                    #-0.043e-3 * {'lhcb1': 1, 'lhcb2': -1}[line_name], # set separation
                    sep_v_ip8 * {'lhcb1': 1, 'lhcb2': -1}[line_name],
                    rtol=0, atol=1e-7)

        if sep_v_ip5 is not None:
            xo.assert_allclose(tw['y', 'ip5'],
                    #-0.043e-3 * {'lhcb1': 1, 'lhcb2': -1}[line_name], # set separation
                    sep_v_ip5 * {'lhcb1': 1, 'lhcb2': -1}[line_name],
                    rtol=0, atol=1e-7)
        xo.assert_allclose(tw['x', 'ip5'], 0, rtol=0, atol=1e-7)


        # Check crossing angles
        xo.assert_allclose(tw['px', 'ip5'],
                170e-6 * {'lhcb1': 1, 'lhcb2': -1}[line_name], rtol=0, atol=0.5e-6)
        xo.assert_allclose(tw['py', 'ip5'], 0, rtol=0, atol=0.5e-6)
        xo.assert_allclose(tw['px', 'ip1'], 0, rtol=0, atol=0.5e-6)
        xo.assert_allclose(tw['py', 'ip1'], 170e-6, rtol=0, atol=0.5e-6)

        xo.assert_allclose(tw['px', 'ip2'], 0, rtol=0, atol=0.5e-6)
        xo.assert_allclose(tw['py', 'ip2'], 100e-6 , rtol=0, atol=0.5e-6) # accounts for spectrometer

        xo.assert_allclose(tw['px', 'ip8'],
                35e-6* {'lhcb1': 1, 'lhcb2': -1}[line_name], rtol=0, atol=0.5e-6) # accounts for spectrometer
        xo.assert_allclose(tw['py', 'ip8'], -2e-6, rtol=0, atol=0.5e-6) # small effect from spectrometer (titled)

        xo.assert_allclose(tw['betx', 'ip1'], 50e-2, rtol=2e-2, atol=0) # beta beating coming from on_disp
        xo.assert_allclose(tw['bety', 'ip1'], 50e-2, rtol=3e-2, atol=0)
        xo.assert_allclose(tw['betx', 'ip5'], 50e-2, rtol=2e-2, atol=0)
        xo.assert_allclose(tw['bety', 'ip5'], 50e-2, rtol=2e-2, atol=0)

        xo.assert_allclose(tw['betx', 'ip2'], 50e-2, rtol=4e-2, atol=0)
        xo.assert_allclose(tw['bety', 'ip2'], 50e-2, rtol=3e-2, atol=0)

        xo.assert_allclose(tw['betx', 'ip8'], 1.5, rtol=3e-2, atol=0)
        xo.assert_allclose(tw['bety', 'ip8'], 1.5, rtol=2e-2, atol=0)

        # Check crab cavities
        z_crab_test = 1e-2
        phi_crab_1 = ((
            collider[line_name].twiss(method='4d', zeta0=z_crab_test)['x', 'ip1']
        - collider[line_name].twiss(method='4d', zeta0=-z_crab_test)['x', 'ip1'])
        / 2 / z_crab_test)

        phi_crab_5 = ((
            collider[line_name].twiss(method='4d', zeta0=z_crab_test)['y', 'ip5']
        - collider[line_name].twiss(method='4d', zeta0=-z_crab_test)['y', 'ip5'])
        / 2 / z_crab_test)

        xo.assert_allclose(phi_crab_1, 0 * {'lhcb1': 1, 'lhcb2': -1}[line_name],
                        rtol=0, atol=5e-7)
        xo.assert_allclose(phi_crab_5, 0, rtol=0, atol=5e-7)

        # Check one octupole strength
        if line_name == 'lhcb1':
            xo.assert_allclose(collider['lhcb1']['mo.33l4.b1'].knl[3], 2.2169*250/235,
                          rtol=1e-3, atol=0)
        elif line_name == 'lhcb2':
            xo.assert_allclose(collider['lhcb2']['mo.33r4.b2'].knl[3], 2.2169*250/235,
                            rtol=1e-3, atol=0)

filling_scheme = {"beam2": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "beam1": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}