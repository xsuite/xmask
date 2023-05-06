import xtrack as xt

import lumi

collider = xt.Multiline.from_json('../hllhc15_collision/collider_02_tuned_bb_off.json')
collider.build_trackers()

tw = collider.twiss(lines=['lhcb1', 'lhcb2'])
assert tuple(tw._line_names) == ('lhcb1', 'lhcb2')
assert 'mqs.23r2.b1' in tw.lhcb1.name
assert 'mqs.23l4.b2' in tw.lhcb2.name
assert tw.lhcb1['s', 'ip5'] < tw.lhcb1['s', 'ip6']
assert tw.lhcb2['s', 'ip5'] > tw.lhcb2['s', 'ip6']



collider.match(
    lines=['lhcb1', 'lhcb2'],
    vary=[
        xt.Vary('kqtf.b1', step=1e-8),
        xt.Vary('kqtd.b1', step=1e-8),
        xt.Vary('kqtf.b2', step=1e-8),
        xt.Vary('kqtd.b2', step=1e-8),
    ],
    targets = [
        xt.Target('qx', line='lhcb1', value=62.317, tol=1e-4),
        xt.Target('qy', line='lhcb1', value=60.327, tol=1e-4),
        xt.Target('qx', line='lhcb2', value=62.315, tol=1e-4),
        xt.Target('qy', line='lhcb2', value=60.325, tol=1e-4)
        ]
    )


