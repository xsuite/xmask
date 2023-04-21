import numpy as np

import xtrack as xt
import xpart as xp

from copy import deepcopy

import lumi

twiss_b1 = xt.twiss.TwissTable(
    data=dict(
        s=np.array([0, 15000, 27000]),
        name=np.array(['ip1', 'ip5', 'end_ring']),
        betx=np.array([55.0e-2, 55.0e-2, 55.0e-2]),
        bety=np.array([55.0e-2, 55.0e-2, 55.0e-2]),
        alfx=np.array([0.0, 0.0, 0.0]),
        alfy=np.array([0.0, 0.0, 0.0]),
        dx=np.array([0.0, 0.0, 0.0]),
        dpx=np.array([0.0, 0.0, 0.0]),
        dy=np.array([0.0, 0.0, 0.0]),
        dpy=np.array([0.0, 0.0, 0.0]),
        x=np.array([0.0, 0.0, 0.0]),
        px=np.array([0, 285e-6/2, 0]),
        y=np.array([0.0, 0.0, 0.0]),
        py=np.array([285e-6/2, 0, 285e-6/2]),
    ))

twiss_b2 = xt.twiss.TwissTable(data=deepcopy(twiss_b1._data))
twiss_b2.px *= -1
twiss_b2.py *= -1

twiss_b1.T_rev0=8.892446333483924e-05
twiss_b1.particle_on_co=xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, p0c=7e12)

twiss_b2.T_rev0=8.892446333483924e-05
twiss_b2.particle_on_co=xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, p0c=7e12)

n_colliding_bunches = 2808
num_particles_per_bunch = 1.15e11
nemitt_x = 3.75e-6
nemitt_y = 3.75e-6
sigma_z = 0.0755

ll_ip1 = lumi.luminosity_from_twiss(
    n_colliding_bunches=n_colliding_bunches,
    num_particles_per_bunch=num_particles_per_bunch,
    ip_name='ip1',
    nemitt_x=nemitt_x,
    nemitt_y=nemitt_y,
    sigma_z=sigma_z,
    twiss_b1=twiss_b1,
    twiss_b2=twiss_b2)

ll_ip5 = lumi.luminosity_from_twiss(
    n_colliding_bunches=n_colliding_bunches,
    num_particles_per_bunch=num_particles_per_bunch,
    ip_name='ip5',
    nemitt_x=nemitt_x,
    nemitt_y=nemitt_y,
    sigma_z=sigma_z,
    twiss_b1=twiss_b1,
    twiss_b2=twiss_b2)
