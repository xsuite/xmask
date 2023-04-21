import xtrack as xt

import lumi

collider = xt.Multiline.from_json('../hllhc15_collision/collider_02_tuned_bb_off.json')
collider.build_trackers()

n_colliding_bunches = 2808
num_particles_per_bunch = 1.15e11
ip_name = 'ip8'
nemitt_x = 3.75e-6
nemitt_y = 3.75e-6
sigma_z = 0.0755

twiss_b1 = collider.lhcb1.twiss()
twiss_b2 = collider.lhcb2.twiss().reverse()

ll = lumi.luminosity_from_twiss(
    n_colliding_bunches=n_colliding_bunches,
    num_particles_per_bunch=num_particles_per_bunch,
    ip_name=ip_name,
    nemitt_x=nemitt_x,
    nemitt_y=nemitt_y,
    sigma_z=sigma_z,
    twiss_b1=twiss_b1,
    twiss_b2=twiss_b2)

def _lumi_to_match(tw):
    return lumi.luminosity_from_twiss(
        n_colliding_bunches=n_colliding_bunches,
        num_particles_per_bunch=num_particles_per_bunch,
        ip_name=ip_name,
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
        sigma_z=sigma_z,
        twiss_b1=tw['lhcb1'],
        twiss_b2=tw['lhcb2'])

xt.match.match_line(
    verbose=True,
    line=collider,
    lines=['lhcb1', 'lhcb2'],
    vary=[xt.Vary('on_sep8', step=1e-6)],
    targets=[xt.Target(_lumi_to_match, 2e32, tol=1e30)])
