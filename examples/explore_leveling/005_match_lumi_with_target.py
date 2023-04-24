import numpy as np

import xtrack as xt

import lumi

collider = xt.Multiline.from_json('../hllhc15_collision/collider_02_tuned_bb_off.json')
collider.build_trackers()




class TargetLuminosity(xt.Target):

    def __init__(self, ip_name, luminosity, tol, num_colliding_bunches,
                 num_particles_per_bunch, nemitt_x, nemitt_y, sigma_z,
                 crab=None):

        xt.Target.__init__(self, self.compute_luminosity, luminosity, tol=tol)

        self.ip_name = ip_name
        self.num_colliding_bunches = num_colliding_bunches
        self.num_particles_per_bunch = num_particles_per_bunch
        self.nemitt_x = nemitt_x
        self.nemitt_y = nemitt_y
        self.sigma_z = sigma_z
        self.crab = crab
        self.scale = 1

    def compute_luminosity(self, tw):
        assert len(tw._line_names) == 2
        return lumi.luminosity_from_twiss(
            n_colliding_bunches=self.num_colliding_bunches,
            num_particles_per_bunch=self.num_particles_per_bunch,
            ip_name=self.ip_name,
            nemitt_x=self.nemitt_x,
            nemitt_y=self.nemitt_y,
            sigma_z=self.sigma_z,
            twiss_b1=tw[tw._line_names[0]],
            twiss_b2=tw[tw._line_names[1]],
            crab=self.crab)

class TargetSeparationOrthogonalToCrossing(xt.Target):

    def __init__(self, ip_name):
        xt.Target.__init__(self, tar=self.projection, value=0, tol=1e-6)
        self.ip_name = ip_name

    def projection(self, tw):
        assert len(tw._line_names) == 2
        ip_name = self.ip_name

        twb1 = tw[tw._line_names[0]]
        twb2 = tw[tw._line_names[1]].reverse()

        diff_px = twb1['px', ip_name] - twb2['px', ip_name]
        diff_py = twb1['py', ip_name] - twb2['py', ip_name]

        diff_p_mod = np.sqrt(diff_px**2 + diff_py**2)

        diff_x = twb1['x', ip_name] - twb2['x', ip_name]
        diff_y = twb1['y', ip_name] - twb2['y', ip_name]

        diff_r_mod = np.sqrt(diff_x**2 + diff_y**2)

        return (diff_x*diff_px + diff_y*diff_py)/(diff_p_mod*diff_r_mod)

# Move to external vertical crossing

collider.vars['phi_ir8'] = 90.

print(f'Knobs before matching: on_sep8h = {collider.vars["on_sep8h"]._value} '
        f'on_sep8v = {collider.vars["on_sep8v"]._value}')

num_colliding_bunches = 2808
num_particles_per_bunch = 1.15e11
nemitt_x = 3.75e-6
nemitt_y = 3.75e-6
sigma_z = 0.0755

collider.match(
    verbose=True,
    lines=['lhcb1', 'lhcb2'],
    vary=[xt.Vary('on_sep8h', step=1e-4),
          xt.Vary('on_sep8v', step=1e-4),],
    targets=[TargetLuminosity(ip_name='ip8', luminosity=2e32, tol=1e30,
                                num_colliding_bunches=num_colliding_bunches,
                                num_particles_per_bunch=num_particles_per_bunch,
                                nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                                sigma_z=sigma_z, crab=False),
             TargetSeparationOrthogonalToCrossing(ip_name='ip8')],
)

print (f'Knobs after matching: on_sep8h = {collider.vars["on_sep8h"]._value} '
        f'on_sep8v = {collider.vars["on_sep8v"]._value}')

tw_after_match = collider.twiss(lines=['lhcb1', 'lhcb2'])
ll_after_match = lumi.luminosity_from_twiss(
    n_colliding_bunches=num_colliding_bunches,
    num_particles_per_bunch=num_particles_per_bunch,
    ip_name='ip8',
    nemitt_x=nemitt_x,
    nemitt_y=nemitt_y,
    sigma_z=sigma_z,
    twiss_b1=tw_after_match['lhcb1'],
    twiss_b2=tw_after_match['lhcb2'],
    crab=False)

assert np.isclose(ll_after_match, 2e32, rtol=1e-2, atol=0)

# Check orthogonality
tw_b1 = tw_after_match['lhcb1']
tw_b4 = tw_after_match['lhcb2']
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





