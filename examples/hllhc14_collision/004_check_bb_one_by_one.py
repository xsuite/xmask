import json
import xobjects as xo
import xtrack as xt
import xfields as xf
import numpy as np
from scipy.stats import norm

# Reference constant charge slicing
# from  https://github.com/giadarol/WeakStrong/blob/master/slicing.py
def _get_z_centroids(ho_slices, sigmaz):
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

# Import the weak and strong lines
verbose = True
with open('collider_02_bb_on.json', 'r') as fid:
    collider = xt.Multiline.from_dict(json.load(fid))

collider.build_trackers()

bb_ip_n_list = [1, 2, 5, 8]
num_long_range_encounters_per_side = [25, 20, 25, 20]
nemitt_x = 2e-6
nemitt_y = 3e-6
harmonic_number = 35640
bunch_spacing_buckets = 10
sigmaz = 0.076
num_slices_head_on = 11



ip = 5 # will be parametrized by pytest
num_lr_per_side = 25 # will be parametrized by pytest
name_weak = 'lhcb1' # will be parametrized by pytest
name_strong = 'lhcb2' # will be parametrized by pytest
sorting = {'l': -1 , 'r': 1} # will be parametrized by pytest

# ip = 5 # will be parametrized by pytest
# num_lr_per_side = 25 # will be parametrized by pytest
# name_weak = 'lhcb2' # will be parametrized by pytest
# name_strong = 'lhcb1' # will be parametrized by pytest
# sorting = {'l': 1 , 'r': -1} # will be parametrized by pytest

# The bb lenses are setup based on the twiss taken with the bb off
print('Twiss(es) (with bb off)')
with xt.tracker._temp_knobs(collider, knobs={'beambeam_scale': 0}):
    tw_weak = collider[name_weak].twiss()
    tw_strong = collider[name_strong].twiss(reverse=True)

# Survey starting from ip
print('Survey(s) (starting from ip)')
survey_weak = collider[name_weak].survey(element0=f'ip{ip}')
survey_strong = collider[name_strong].survey(element0=f'ip{ip}', reverse=True)
beta0_strong = collider[name_strong].particle_ref.beta0[0]
gamma0_strong = collider[name_strong].particle_ref.gamma0[0]


bunch_spacing_ds = tw_weak.circumference/harmonic_number*bunch_spacing_buckets

# Check lr encounters
for side in ['l', 'r']:
    for iele in range(num_lr_per_side):
        nn_weak = f'bb_lr.{side}{ip}b{name_weak[-1]}_{iele+1:02d}'
        nn_strong = f'bb_lr.{side}{ip}b{name_strong[-1]}_{iele+1:02d}'

        assert nn_weak in tw_weak.name
        assert nn_strong in tw_strong.name

        ee_weak = collider[name_weak][nn_weak]

        assert isinstance(ee_weak, xf.BeamBeamBiGaussian2D)

        expected_sigma_x = np.sqrt(tw_strong[nn_strong, 'betx']
                                * nemitt_x/beta0_strong/gamma0_strong)
        expected_sigma_y = np.sqrt(tw_strong[nn_strong, 'bety']
                                * nemitt_y/beta0_strong/gamma0_strong)

        # Beam sizes
        assert np.isclose(ee_weak.other_beam_Sigma_11, expected_sigma_x**2,
                          atol=0, rtol=1e-6)
        assert np.isclose(ee_weak.other_beam_Sigma_33, expected_sigma_y**2,
                          atol=0, rtol=1e-6)

        # Orbit
        assert np.isclose(ee_weak.ref_shift_x, tw_weak[nn_weak, 'x'],
                          rtol=0, atol=1e-4 * expected_sigma_x)
        assert np.isclose(ee_weak.ref_shift_y, tw_weak[nn_weak, 'y'],
                            rtol=0, atol=1e-4 * expected_sigma_y)

        # Separation
        assert np.isclose(ee_weak.other_beam_shift_x,
                    tw_strong[nn_strong, 'x'] - tw_weak[nn_weak, 'x']
                    + survey_strong[nn_strong, 'X'] - survey_weak[nn_weak, 'X'],
                    rtol=0, atol=1e-4 * expected_sigma_x)

        assert np.isclose(ee_weak.other_beam_shift_y,
                            tw_strong[nn_strong, 'y'] - tw_weak[nn_weak, 'y']
                            + survey_strong[nn_strong, 'Y'] - survey_weak[nn_weak, 'Y'],
                            rtol=0, atol=1e-4 * expected_sigma_y)

        # s position
        assert np.isclose(tw_weak[nn_weak, 's'] - tw_weak[f'ip{ip}', 's'],
                          bunch_spacing_ds/2 * (iele+1) * sorting[side],
                          rtol=0, atol=10e-6)


# Check head on encounters

# Quick check on _get_z_centroids
assert np.isclose(np.mean(_get_z_centroids(100000, 5.)**2), 5**2,
                          rtol=0, atol=52e-4)
assert np.isclose(np.mean(_get_z_centroids(100000, 5.)), 0,
                          rtol=0, atol=1e-10)

z_centroids = _get_z_centroids(num_slices_head_on, sigmaz)
assert len(z_centroids) == num_slices_head_on

assert num_slices_head_on % 2 == 1
for ii, zz in list(zip(range(-(num_slices_head_on - 1) // 2,
                       (num_slices_head_on - 1) // 2 + 1),
                  z_centroids)):

    if ii == 0:
        side = 'c'
    elif ii < 0:
        side = 'l' if sorting['l'] == -1 else 'r'
    else:
        side = 'r' if sorting['r'] == 1 else 'l'

    nn_weak = f'bb_ho.{side}{ip}b{name_weak[-1]}_{int(abs(ii)):02d}'
    nn_strong = f'bb_ho.{side}{ip}b{name_strong[-1]}_{int(abs(ii)):02d}'

    ee_weak = collider[name_weak][nn_weak]

    assert isinstance(ee_weak, xf.BeamBeamBiGaussian3D)
    assert ee_weak.num_slices_other_beam == 1
    assert ee_weak.slices_other_beam_zeta_center[0] == 0

    # s position
    expected_s = zz / 2
    assert np.isclose(tw_weak[nn_weak, 's'] - tw_weak[f'ip{ip}', 's'],
                      expected_s, atol=10e-6, rtol=0)

    # Beam sizes
    expected_sigma_x = np.sqrt(tw_strong[nn_strong, 'betx']
                            * nemitt_x/beta0_strong/gamma0_strong)
    expected_sigma_y = np.sqrt(tw_strong[nn_strong, 'bety']
                            * nemitt_y/beta0_strong/gamma0_strong)

    assert np.isclose(ee_weak.slices_other_beam_Sigma_11[0], expected_sigma_x**2,
                        atol=0, rtol=1e-6)
    assert np.isclose(ee_weak.slices_other_beam_Sigma_33[0], expected_sigma_y**2,
                        atol=0, rtol=1e-6)

    expected_sigma_px = np.sqrt(tw_strong[nn_strong, 'gamx']
                                * nemitt_x/beta0_strong/gamma0_strong)
    expected_sigma_py = np.sqrt(tw_strong[nn_strong, 'gamy']
                                * nemitt_y/beta0_strong/gamma0_strong)
    assert np.isclose(ee_weak.slices_other_beam_Sigma_22[0], expected_sigma_px**2,
                      atol=0, rtol=1e-6)
    assert np.isclose(ee_weak.slices_other_beam_Sigma_44[0], expected_sigma_py**2,
                      atol=0, rtol=1e-6)

    expected_sigma_xpx = -(tw_strong[nn_strong, 'alfx']
                            * nemitt_x / beta0_strong / gamma0_strong)
    expected_sigma_ypy = -(tw_strong[nn_strong, 'alfy']
                            * nemitt_y / beta0_strong / gamma0_strong)
    assert np.isclose(ee_weak.slices_other_beam_Sigma_12[0], expected_sigma_xpx,
                        atol=0, rtol=1e-5)
    assert np.isclose(ee_weak.slices_other_beam_Sigma_34[0], expected_sigma_ypy,
                        atol=0, rtol=1e-5)


