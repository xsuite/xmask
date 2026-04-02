import xtrack as xt
import xobjects as xo
import xfields as xf
from itertools import product
import numpy as np
import pandas as pd

label = 'thin'

if label == 'thin':
    lhc = xt.load("lhc_thin_test_04_tuned_and_leveled_bb_on.json")
elif label == 'thick':
    lhc = xt.load("lhc_thick_test_04_tuned_and_leveled_bb_on.json")
else:
    raise ValueError



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

    bunch_spacing_ds = (tw_weak.circumference / harmonic_number
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
                * tw_strong.circumference / (2 * np.pi * harmonic_number)
                * np.sin(2 * np.pi * zz
                        * harmonic_number / tw_strong.circumference)),
            rtol=0, atol=1e-6) # Not the cleanest, to be investigated
        # print(f"nn_weak: {nn_weak}, nn_strong: {nn_strong}")
        # print(f"shift_x: {ee_weak.other_beam_shift_x}, expected shift_x: {(tw_strong['x', nn_strong] - tw_weak['x', nn_weak] + survey_strong['X', nn_strong] - survey_weak['X', nn_weak] - phi_crab_x * tw_strong.circumference / (2 * np.pi * harmonic_number) * np.sin(2 * np.pi * zz * harmonic_number / tw_strong.circumference))}")
        # print(f"x_strong: {tw_strong['x', nn_strong]}, x_weak: {tw_weak['x', nn_weak]}, survey_strong: {survey_strong['X', nn_strong]}, survey_weak: {survey_weak['X', nn_weak]}, phi_crab_x: {phi_crab_x}, zz: {zz}")

        xo.assert_allclose(ee_weak.other_beam_shift_y,
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
        if ip_n == 8:
            pass # TODO: tilted crossing, to be checked differently
        elif np.abs(tw_weak['px', f'ip{ip_n}']) < 1e-6:
            # Vertical crossing
            xo.assert_allclose(ee_weak.alpha, np.pi/2, atol=5e-3, rtol=0)
            xo.assert_allclose(
                2*ee_weak.phi,
                tw_weak['py', f'ip{ip_n}'] - tw_strong['py', f'ip{ip_n}'],
                atol=2e-7, rtol=0)
        else:
            # Horizontal crossing
            xo.assert_allclose(ee_weak.alpha,
                (-15e-3 if ip_n==8 else 0)*{'b1': 1, 'b2': -1}[name_weak],
                atol=5e-3, rtol=0)
            xo.assert_allclose(
                2*ee_weak.phi,
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

    fp_polar_with_rescale = lhc[line_name].get_footprint(
        nemitt_x=2.5e-6, nemitt_y=2.5e-6,
        linear_rescale_on_knobs=[
            xt.LinearRescale(knob_name='beambeam_scale', v0=0.0, dv=0.1)]
        )
