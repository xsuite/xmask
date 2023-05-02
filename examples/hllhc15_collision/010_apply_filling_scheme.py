import numpy as np

import xtrack as xt

collider = xt.Multiline.from_json('./collider_04_tuned_and_leveled_bb_on.json')
collider.build_trackers()

filling_pattern_b1 = np.zeros(3564, dtype=int)
filling_pattern_b2 = np.zeros(3564, dtype=int)

# Fill 50 bunches around bunch 500
filling_pattern_b1[500-25:500+25] = 1

