import numpy as np
from scipy.constants import c as clight

import xtrack as xt
import xmask as xm


collider = xt.Environment.from_json('../hllhc15_collision/collider_02_tuned_bb_off.json')
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

# Correct closed orbit
for line_name in ['lhcb1', 'lhcb2']:
    xm.machine_tuning(line=collider[line_name],
        enable_closed_orbit_correction=True,
        enable_linear_coupling_correction=False,
        enable_tune_correction=False,
        enable_chromaticity_correction=False,
        knob_names=[],
        targets=None,
        line_co_ref=collider[line_name+'_co_ref'],
        co_corr_config=f'../hllhc15_collision/corr_co_{line_name}.json')

tw_after_orbit_correction = collider.twiss(lines=['lhcb1', 'lhcb2'])

print(f'Knobs before matching: on_sep8h = {collider.vars["on_sep8h"]._value} '
        f'on_sep8v = {collider.vars["on_sep8v"]._value}')

# Leveling assuming ideal behavior of the knobs
knob_values_before_ideal_matching = {
    'on_sep8h': collider.vars['on_sep8h']._value,
    'on_sep8v': collider.vars['on_sep8v']._value,
}
tw0 = collider.twiss(lines=['lhcb1', 'lhcb2'])
opt = collider.match(
    solver_options={'n_bisections': 3, 'min_step': 1e-5, 'n_steps_max': 200},
    start=['e.ds.l8.b1', 's.ds.r8.b2'],
    end=['s.ds.r8.b1', 'e.ds.l8.b2'],
    init=tw0, init_at=xt.START,
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

print (f'Knobs after ideal matching: on_sep8h = {collider.vars["on_sep8h"]._value} '
        f'on_sep8v = {collider.vars["on_sep8v"]._value}')

# Reset knobs
collider.vars['on_sep8h'] = knob_values_before_ideal_matching['on_sep8h']
collider.vars['on_sep8v'] = knob_values_before_ideal_matching['on_sep8v']

# Leveling with crossing angle and bump rematching
tw0 = collider.twiss(lines=['lhcb1', 'lhcb2'])
collider.match(
    solver_options={'n_bisections': 3, 'min_step': 0, 'n_steps_max': 200},
    lines=['lhcb1', 'lhcb2'],
    start=['e.ds.l8.b1', 's.ds.r8.b2'],
    end=['s.ds.r8.b1', 'e.ds.l8.b2'],
    init=tw0, init_at=xt.START,
    targets=[
        # Luminosity
        xt.TargetLuminosity(
            ip_name='ip8', luminosity=2e14, tol=1e12,
            f_rev=f_rev, num_colliding_bunches=num_colliding_bunches,
            num_particles_per_bunch=num_particles_per_bunch, sigma_z=sigma_z,
            nemitt_x=nemitt_x, nemitt_y=nemitt_y, crab=False),
        # Separation plane inclination
        xt.TargetSeparationOrthogonalToCrossing(ip_name='ip8'),
        # Preserve crossing angle
        xt.TargetList(['px', 'py'], at='ip8', line='lhcb1', value=tw0, tol=1e-7, scale=1e3),
        xt.TargetList(['px', 'py'], at='ip8', line='lhcb2', value=tw0, tol=1e-7, scale=1e3),
        # Close the bumps
        xt.TargetList(['x', 'y'], at='s.ds.r8.b1', line='lhcb1', value=tw0, tol=1e-5, scale=1),
        xt.TargetList(['px', 'py'], at='s.ds.r8.b1', line='lhcb1', value=tw0, tol=1e-5, scale=1e3),
        xt.TargetList(['x', 'y'], at='e.ds.l8.b2', line='lhcb2', value=tw0, tol=1e-5, scale=1),
        xt.TargetList(['px', 'py'], at='e.ds.l8.b2', line='lhcb2', value=tw0, tol=1e-5, scale=1e3),
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

print (f'Knobs after full matching: on_sep8h = {collider.vars["on_sep8h"]._value} '
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
tw0 = collider.twiss(lines=['lhcb1', 'lhcb2'])
collider.match(
    lines=['lhcb1', 'lhcb2'],
    start=['e.ds.l2.b1', 's.ds.r2.b2'],
    end=['s.ds.r2.b1', 'e.ds.l2.b2'],
    init=tw0, init_at=xt.START,
    targets=[
        xt.TargetSeparation(ip_name='ip2', separation_norm=3, plane='x', tol=1e-4,
                         nemitt_x=nemitt_x, nemitt_y=nemitt_y),
        # Preserve crossing angle
        xt.TargetList(['px', 'py'], at='ip2', line='lhcb1', value=tw0, tol=1e-7, scale=1e3),
        xt.TargetList(['px', 'py'], at='ip2', line='lhcb2', value=tw0, tol=1e-7, scale=1e3),
        # Close the bumps
        xt.TargetList(['x', 'y'], at='s.ds.r2.b1', line='lhcb1', value=tw0, tol=1e-5, scale=1),
        xt.TargetList(['px', 'py'], at='s.ds.r2.b1', line='lhcb1', value=tw0, tol=1e-5, scale=1e3),
        xt.TargetList(['x', 'y'], at='e.ds.l2.b2', line='lhcb2', value=tw0, tol=1e-5, scale=1),
        xt.TargetList(['px', 'py'], at='e.ds.l2.b2', line='lhcb2', value=tw0, tol=1e-5, scale=1e3),
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






