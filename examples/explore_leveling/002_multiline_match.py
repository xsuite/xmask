import xtrack as xt

import lumi

collider = xt.Multiline.from_json('../hllhc15_collision/collider_02_tuned_bb_off.json')
collider.build_trackers()

tw = collider.twiss(lines=['lhcb1', 'lhcb2'])

collider.lhcb1.match(
    vary=[
        xt.Vary('kqtf.b1', step=1e-8),
        xt.Vary('kqtd.b1', step=1e-8),
    ],
    targets = [
        xt.Target('qx', 62.315, tol=1e-4),
        xt.Target('qy', 60.325, tol=1e-4)])

xt.match.match_line(
    line=collider,
    lines=['lhcb1', 'lhcb2'],
    vary=[
        xt.Vary('kqtf.b1', step=1e-8),
        xt.Vary('kqtd.b1', step=1e-8),
        xt.Vary('kqtf.b2', step=1e-8),
        xt.Vary('kqtd.b2', step=1e-8),
    ],
    targets = [
        xt.Target(lambda tw: tw['lhcb1'].qx, 62.311, tol=1e-4),
        xt.Target(lambda tw: tw['lhcb1'].qy, 60.321, tol=1e-4),
        xt.Target(lambda tw: tw['lhcb2'].qx, 62.312, tol=1e-4),
        xt.Target(lambda tw: tw['lhcb2'].qy, 60.322, tol=1e-4)
        ]
    )