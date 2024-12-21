import matplotlib.pyplot as plt
import xtrack as xt

collider = xt.Environment.from_json('./collider_04_tuned_and_leveled_bb_on.json')
collider.build_trackers()

fp = collider['lhcb1'].get_footprint(
    nemitt_x=1.65e-6, nemitt_y=1.65e-6)

fp.plot()

plt.show()