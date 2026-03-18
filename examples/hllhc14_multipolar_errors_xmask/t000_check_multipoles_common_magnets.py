import xtrack as xt
import xobjects as xo
from load_wise import assert_are_same_multipoles_b1_b2

ele_b1 = xt.Multipole(knl=[0.001, 0.1, 0.2, 3], ksl=[0.002, 0.3, 0.4, 5])
ele_b2 = xt.Multipole(knl=[0.001, -0.1, 0.2, -3], ksl=[-0.002, 0.3, -0.4, 5])

env = xt.Environment()
env.elements['ele_b1'] = ele_b1
env.elements['ele_b2'] = ele_b2
env.elements['ip_side'] = xt.Marker()
env.elements['arc_side'] = xt.Marker()

line_b1 = env.new_line(components=['ip_side', 'ele_b1', 'arc_side'])
line_b2 = env.new_line(components=['arc_side', 'ele_b2', 'ip_side'])

line_b1.set_particle_ref(p0c=7000e9)
line_b2.set_particle_ref(p0c=7000e9)

# Twiss with output in the same reference system
tw1 = line_b1.twiss(betx=1, bety=1, x=0.01, y=0.02)
tw2 = line_b2.twiss(betx=1, bety=1, x=0.01, y=0.02, init_at='ip_side',
                    reverse=True)

# In [24]: tw1.cols['px py']
# Out[24]:
# TwissTable: 4 rows, 3 cols
# name                  px            py
# ip_side                0             0
# ele_b1                 0             0
# arc_side      0.00411383    0.00696983
# _end_point    0.00411383    0.00696983

# In [25]: tw2.cols['px py']
# Out[25]:
# TwissTable: 4 rows, 3 cols
# name                  px            py
# ip_side                0            -0
# ele_b2                 0            -0
# arc_side     -0.00411383   -0.00696983
# _end_point   -0.00411383   -0.00696983

xo.assert_allclose(tw1.px, -tw2.px, rtol=1e-12, atol=1e-12)
xo.assert_allclose(tw1.py, -tw2.py, rtol=1e-12, atol=1e-12)

assert_are_same_multipoles_b1_b2(ele_b1, ele_b2, rtol=1e-12, atol=1e-12)