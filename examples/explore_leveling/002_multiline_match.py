import xtrack as xt

import lumi

collider = xt.Multiline.from_json('../hllhc15_collision/collider_02_tuned_bb_off.json')
collider.build_trackers()

tw = collider.twiss(lines=['lhcb1', 'lhcb2'])

collider.match(
    lines=['lhcb1', 'lhcb2'],
    vary=[
        xt.Vary('kqtf.b1', step=1e-8),
        xt.Vary('kqtd.b1', step=1e-8),
        xt.Vary('kqtf.b2', step=1e-8),
        xt.Vary('kqtd.b2', step=1e-8),
    ],
    targets = [
        xt.Target('qx', line='lhcb1', value=62.31, tol=1e-4),
        xt.Target('qy', line='lhcb1', value=60.32, tol=1e-4),
        xt.Target('qx', line='lhcb2', value=62.315, tol=1e-4),
        xt.Target('qy', line='lhcb2', value=60.325, tol=1e-4)
        ]
    )