import numpy as np

import xtrack as xt

import lumi

collider = xt.Multiline.from_json('../hllhc15_collision/collider_02_tuned_bb_off.json')
collider.build_trackers()


class TargetLuminosity(xt.Target):

    def __init__(self, ip_name, luminosity, tol, num_colliding_bunches,
                 num_particles_per_bunch, nemitt_x, nemitt_y, sigma_z,
                 crab=None, scale=1):

        xt.Target.__init__(self, self.compute_luminosity, luminosity, tol=tol)

        self.ip_name = ip_name
        self.num_colliding_bunches = num_colliding_bunches
        self.num_particles_per_bunch = num_particles_per_bunch
        self.nemitt_x = nemitt_x
        self.nemitt_y = nemitt_y
        self.sigma_z = sigma_z
        self.crab = crab
        self.scale = scale

    def compute_luminosity(self, tw):
        return lumi.luminosity_from_twiss(
            n_colliding_bunches=self.num_colliding_bunches,
            num_particles_per_bunch=self.num_particles_per_bunch,
            ip_name=self.ip_name,
            nemitt_x=self.nemitt_x,
            nemitt_y=self.nemitt_y,
            sigma_z=self.sigma_z,
            twiss_b1=tw['lhcb1'],
            twiss_b2=tw['lhcb2'],
            crab=self.crab)

num_colliding_bunches = 2808
num_particles_per_bunch = 1.15e11
nemitt_x = 3.75e-6
nemitt_y = 3.75e-6
sigma_z = 0.0755

collider.match(
    lines=['lhcb1', 'lhcb2'],
    vary=[xt.Vary('on_sep8', step=1e-4)],
    targets=[TargetLuminosity(ip_name='ip8', luminosity=2e32, tol=1e30,
                                num_colliding_bunches=num_colliding_bunches,
                                num_particles_per_bunch=num_particles_per_bunch,
                                nemitt_x=nemitt_x, nemitt_y=nemitt_y,
                                sigma_z=sigma_z, crab=False)])

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
