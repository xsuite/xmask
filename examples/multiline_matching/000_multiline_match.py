import numpy as np
import xtrack as xt

collider = xt.Environment.from_json('../hllhc15_collision/collider_02_tuned_bb_off.json')
collider.build_trackers()

tw = collider.twiss(lines=['lhcb1', 'lhcb2'])
assert tuple(tw._line_names) == ('lhcb1', 'lhcb2')
assert 'mqs.23r2.b1' in tw.lhcb1.name
assert 'mqs.23l4.b2' in tw.lhcb2.name
assert tw.lhcb1['s', 'ip5'] < tw.lhcb1['s', 'ip6']
assert tw.lhcb2['s', 'ip5'] > tw.lhcb2['s', 'ip6']
assert np.isclose(tw.lhcb1.qx, 62.31, atol=1e-4, rtol=0)
assert np.isclose(tw.lhcb1.qy, 60.32, atol=1e-4, rtol=0)
assert np.isclose(tw.lhcb2.qx, 62.31, atol=1e-4, rtol=0)
assert np.isclose(tw.lhcb2.qy, 60.32, atol=1e-4, rtol=0)

tw_part = collider.twiss(
    lines=['lhcb1', 'lhcb2'],
    start=['ip5', 'ip6'],
    end=['ip6', 'ip5'],
    init=[tw.lhcb1.get_twiss_init(at_element='ip5'), tw.lhcb2.get_twiss_init(at_element='ip6')]
)

# Add some asserts here

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

tw1 = collider.twiss(lines=['lhcb1', 'lhcb2'])
assert tuple(tw1._line_names) == ('lhcb1', 'lhcb2')
assert 'mqs.23r2.b1' in tw1.lhcb1.name
assert 'mqs.23l4.b2' in tw1.lhcb2.name
assert tw1.lhcb1['s', 'ip5'] < tw1.lhcb1['s', 'ip6']
assert tw1.lhcb2['s', 'ip5'] > tw1.lhcb2['s', 'ip6']
assert np.isclose(tw1.lhcb1.qx, 62.317, atol=1e-4, rtol=0)
assert np.isclose(tw1.lhcb1.qy, 60.327, atol=1e-4, rtol=0)
assert np.isclose(tw1.lhcb2.qx, 62.315, atol=1e-4, rtol=0)
assert np.isclose(tw1.lhcb2.qy, 60.325, atol=1e-4, rtol=0)

# Match bumps in the two likes
tw0 = collider.twiss(lines=['lhcb1', 'lhcb2'])
collider.match(
    lines=['lhcb1', 'lhcb2'],
    start=['mq.33l8.b1', 'mq.22l8.b2'],
    end=['mq.23l8.b1', 'mq.32l8.b2'],
    init=tw0, init_at=xt.START,
    vary=[
        xt.VaryList([
            'acbv30.l8b1', 'acbv28.l8b1', 'acbv26.l8b1', 'acbv24.l8b1'],
            step=1e-10),
        xt.VaryList([
            'acbv29.l8b2', 'acbv27.l8b2', 'acbv25.l8b2', 'acbv23.l8b2'],
            step=1e-10),
    ],
    targets=[
        xt.Target('y', at='mb.b28l8.b1', line='lhcb1', value=3e-3, tol=1e-4, scale=1),
        xt.Target('py', at='mb.b28l8.b1', line='lhcb1', value=0, tol=1e-6, scale=1000),
        xt.Target('y', at='mb.b27l8.b2', line='lhcb2', value=2e-3, tol=1e-4, scale=1),
        xt.Target('py', at='mb.b27l8.b2', line='lhcb2', value=0, tol=1e-6, scale=1000),
        # I want the bump to be closed
        xt.TargetList(['y'], at='mq.23l8.b1', line='lhcb1', value=tw0, tol=1e-6, scale=1),
        xt.TargetList(['py'], at='mq.23l8.b1', line='lhcb1', value=tw0, tol=1e-7, scale=1000),
        xt.TargetList(['y'], at='mq.32l8.b2', line='lhcb2', value=tw0, tol=1e-6, scale=1),
        xt.Target('py', at='mq.32l8.b2', line='lhcb2', value=tw0, tol=1e-10, scale=1000),
    ]
)
tw_bump = collider.twiss(lines=['lhcb1', 'lhcb2'])

tw_before = tw1.lhcb1
assert np.isclose(tw_bump.lhcb1['y', 'mb.b28l8.b1'], 3e-3, atol=1e-4)
assert np.isclose(tw_bump.lhcb1['py', 'mb.b28l8.b1'], 0, atol=1e-6)
assert np.isclose(tw_bump.lhcb1['y', 'mq.23l8.b1'], tw_before['y', 'mq.23l8.b1'], atol=1e-6)
assert np.isclose(tw_bump.lhcb1['py', 'mq.23l8.b1'], tw_before['py', 'mq.23l8.b1'], atol=1e-7)
assert np.isclose(tw_bump.lhcb1['y', 'mq.33l8.b1'], tw_before['y', 'mq.33l8.b1'], atol=1e-6)
assert np.isclose(tw_bump.lhcb1['py', 'mq.33l8.b1'], tw_before['py', 'mq.33l8.b1'], atol=1e-7)

tw_before = tw1.lhcb2
assert np.isclose(tw_bump.lhcb2['y', 'mb.b27l8.b2'], 2e-3, atol=1e-4)
assert np.isclose(tw_bump.lhcb2['py', 'mb.b27l8.b2'], 0, atol=1e-6)
assert np.isclose(tw_bump.lhcb2['y', 'mq.32l8.b2'], tw_before['y', 'mq.33l8.b2'], atol=1e-6)
assert np.isclose(tw_bump.lhcb2['py', 'mq.32l8.b2'], tw_before['py', 'mq.33l8.b2'], atol=1e-7)
assert np.isclose(tw_bump.lhcb2['y', 'mq.22l8.b2'], tw_before['y', 'mq.23l8.b2'], atol=1e-6)
assert np.isclose(tw_bump.lhcb2['py', 'mq.22l8.b2'], tw_before['py', 'mq.23l8.b2'], atol=1e-7)

