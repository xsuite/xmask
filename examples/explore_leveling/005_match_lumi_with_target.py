import numpy as np
from scipy.constants import c as clight

import xtrack as xt
import xmask as xm

collider = xt.Multiline.from_json('../hllhc15_collision/collider_02_tuned_bb_off.json')
collider.build_trackers()

# Move to external vertical crossing
collider.vars['phi_ir8'] = 90.

tw_before_errors = collider.twiss(lines=['lhcb1', 'lhcb2'])

# Introduce dipolar errors in D1 and D2
# collider.lhcb1['mqxa.1l8..5'].knl[0] = +2e-6
# collider.lhcb1['mqxa.1l8..5'].ksl[0] = -3e-6
# collider.lhcb1['mqxa.1r8..5'].knl[0] = -1e-6
# collider.lhcb1['mqxa.1r8..5'].ksl[0] = +4e-6

# collider.lhcb1['mqxa.1r8..5'].knl[1] *= 0.7
# collider.lhcb1['mqxa.1r8..5'].knl[1] *= 1.5
# collider.lhcb1['mqxa.1l8..5'].ksl[1] = 0.001
collider.lhcb1['mqxb.a2r8..5'].knl[1] = collider.lhcb1['mqxb.a2r8..4'].knl[1] * 1.3

# collider.lhcb1['mqxb.a2r8..5'].ksl[1] = 0.001
# collider.lhcb1['mqy.a4l8.b1..1'].knl[1] *= 1.3
# collider.lhcb1['mqy.a4r8.b1..1'].knl[1] *= 0.8
# collider.lhcb1['mqxa.1r8..5'].knl[1] *= 0.7




tw_after_errors = collider.twiss(lines=['lhcb1', 'lhcb2'])

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

num_colliding_bunches = 2808
num_particles_per_bunch = 1.15e11
nemitt_x = 3.75e-6
nemitt_y = 3.75e-6
sigma_z = 0.0755
beta0_b1 = collider.lhcb1.particle_ref.beta0[0]

# Correction assuming ideal behavior of the knobs
knob_values_before_ideal_matching = {
    'on_sep8h': collider.vars['on_sep8h']._value,
    'on_sep8v': collider.vars['on_sep8v']._value,
}

collider.match(
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



print(f'Before ideal matching: px = {tw_after_orbit_correction["lhcb1"]["px", "ip8"]:.3e} ')
print(f'After ideal matching:  px = {tw_after_ideal_lumi_matching["lhcb1"]["px", "ip8"]:.3e} ')
print(f'Before ideal matching: py = {tw_after_orbit_correction["lhcb1"]["py", "ip8"]:.3e} ')
print(f'After ideal matching:  py = {tw_after_ideal_lumi_matching["lhcb1"]["py", "ip8"]:.3e} ')


prrrr

collider.match(
    ele_start=['e.ds.l8.b1', 's.ds.r8.b2'],
    ele_stop=['s.ds.r8.b1', 'e.ds.l8.b2'],
    twiss_init='preserve',
    lines=['lhcb1', 'lhcb2'],
    vary=[
        # Knobs to control the separation
        xt.Vary('on_sep8h', step=1e-4),
        xt.Vary('on_sep8v', step=1e-4),

        # Correctors to preserve crossing angle
        xt.Vary('corr_co_acbyvs4.l8b1', step=1e-7),
        xt.Vary('corr_co_acbyhs4.l8b1', step=1e-7),
        xt.Vary('corr_co_acbyvs4.r8b2', step=1e-7),
        xt.Vary('corr_co_acbyhs4.r8b2', step=1e-7),

        # # Correctors to close the bumps
        # xt.Vary('corr_co_acbyvs4.r8b1', step=1e-7),
        # xt.Vary('corr_co_acbyhs4.r8b1', step=1e-7),
        # xt.Vary('corr_co_acbyvs4.l8b2', step=1e-7),
        # xt.Vary('corr_co_acbyhs4.l8b2', step=1e-7),
        # xt.Vary('corr_co_acbyvs5.r8b1', step=1e-7),
        # xt.Vary('corr_co_acbyhs5.r8b1', step=1e-7),
        # xt.Vary('corr_co_acbcvs5.l8b2', step=1e-7),
        # xt.Vary('corr_co_acbchs5.l8b2', step=1e-7),
        ],
    targets=[
        xt.TargetLuminosity(ip_name='ip8',
                                luminosity=2e32,
                                tol=1e28,
                                f_rev=1/(collider.lhcb1.get_length() /(beta0_b1 * clight)),
                                num_colliding_bunches=num_colliding_bunches,
                                num_particles_per_bunch=num_particles_per_bunch,
                                nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                                sigma_z=sigma_z, crab=False),
        xt.TargetSeparationOrthogonalToCrossing(ip_name='ip8'),
        # Preserve crossing angle
        xt.Target('px', at='ip8', line='lhcb1', value='preserve', tol=1e-7, scale=1e3),
        xt.Target('py', at='ip8', line='lhcb1', value='preserve', tol=1e-7, scale=1e3),
        xt.Target('px', at='ip8', line='lhcb2', value='preserve', tol=1e-7, scale=1e3),
        xt.Target('py', at='ip8', line='lhcb2', value='preserve', tol=1e-7, scale=1e3),
        # # Close the bumps
        # xt.Target('x', at='s.ds.r8.b1', line='lhcb1', value='preserve'),
        # xt.Target('px', at='s.ds.r8.b1', line='lhcb1', value='preserve'),
        # xt.Target('y', at='s.ds.r8.b1', line='lhcb1', value='preserve'),
        # xt.Target('py', at='s.ds.r8.b1', line='lhcb1', value='preserve'),
        # xt.Target('x', at='e.ds.l8.b2', line='lhcb2', value='preserve'),
        # xt.Target('px', at='e.ds.l8.b2', line='lhcb2', value='preserve'),
        # xt.Target('y', at='e.ds.l8.b2', line='lhcb2', value='preserve'),
        # xt.Target('py', at='e.ds.l8.b2', line='lhcb2', value='preserve'),
        ],
)

print (f'Knobs after matching: on_sep8h = {collider.vars["on_sep8h"]._value} '
        f'on_sep8v = {collider.vars["on_sep8v"]._value}')

tw_after_ip8_match = collider.twiss(lines=['lhcb1', 'lhcb2'])
ll_after_match = xt.lumi.luminosity_from_twiss(
    n_colliding_bunches=num_colliding_bunches,
    num_particles_per_bunch=num_particles_per_bunch,
    ip_name='ip8',
    nemitt_x=nemitt_x,
    nemitt_y=nemitt_y,
    sigma_z=sigma_z,
    twiss_b1=tw_after_ip8_match['lhcb1'],
    twiss_b2=tw_after_ip8_match['lhcb2'],
    crab=False)

assert np.isclose(ll_after_match, 2e32, rtol=1e-2, atol=0)

# Check orthogonality
tw_b1 = tw_after_ip8_match['lhcb1']
tw_b4 = tw_after_ip8_match['lhcb2']
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
    verbose=True,
    lines=['lhcb1', 'lhcb2'],
    vary=[xt.Vary('on_sep2', step=1e-4)],
    targets=[
        xt.TargetSeparation(ip_name='ip2', separation_norm=3, plane='x', tol=1e-4,
                         nemitt_x=nemitt_x, nemitt_y=nemitt_y),
    ],
)
print(f'Knobs after matching: on_sep2 = {collider.vars["on_sep2"]._value}')

tw_after_ip2_match = collider.twiss(lines=['lhcb1', 'lhcb2'])

# Check normalized separation
mean_betx = (tw_after_ip2_match['lhcb1']['betx', 'ip2']
                + tw_after_ip2_match['lhcb2']['betx', 'ip2']) / 2
gamma0 = tw_after_ip2_match['lhcb1'].particle_on_co.gamma0[0]
beta0 = tw_after_ip2_match['lhcb1'].particle_on_co.beta0[0]
sigmax = np.sqrt(nemitt_x * mean_betx /gamma0 / beta0)

assert np.isclose(collider.vars['on_sep2']._value/1000, 3*sigmax/2, rtol=1e-3, atol=0)

# 1e-4 per testare

# First match
# range: s.ds - ip
# Target
#     TargetLuminosity,
#     TargetSeparationOrthogonalCrossing
#     px, py b1 @ ip
#     px, py b2 @ ip
# Vary
#     on_seph on_sepv,
#     as many many correctors as needed





