import xtrack as xt
import xobjects as xo

env = xt.load('collider_errors_on_corrections_on.json')

def assert_are_same_multipoles_b1_b2(ele_b1, ele_b2, atol=0, rtol=0):
    knl_b1, ksl_b1 = ele_b1.get_total_knl_ksl()
    knl_b2, ksl_b2 = ele_b2.get_total_knl_ksl()
    knl_b2_check = knl_b2.copy()
    ksl_b2_check = ksl_b2.copy()
    knl_b2_check[1::2] *= -1
    ksl_b2_check[0::2] *= -1

    xo.assert_allclose(knl_b1, knl_b2_check, rtol=rtol, atol=atol)
    xo.assert_allclose(ksl_b1, ksl_b2_check, rtol=rtol, atol=atol)


tt_b1_mqxf = env['lhcb1'].get_table(attr=True).rows['mqx.*']
for nn in tt_b1_mqxf.name:
    if '..fl' in nn or '..fr' in nn:
        continue # skip edge lenses
    if isinstance(env[nn], xt.Marker):
        continue # skip markers
    ele_b1 = env['lhcb1'][nn]
    ele_b2 = env['lhcb2'][nn.replace('/lhcb1', '/lhcb2')]
    assert_are_same_multipoles_b1_b2(ele_b1, ele_b2, rtol=1e-12, atol=1e-12)

