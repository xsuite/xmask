import xtrack as xt
import xmask as xm

# Read config file
with open('config.yaml','r') as fid:
    config = xm.yaml.load(fid)

lhc = xt.load(f'lhc_{config["label"]}_00_prepared.json')
lhc_co_ref = xt.load(f'lhc_co_ref_{config["label"]}.json')

for kk, vv in config['knob_settings'].items():
    lhc[kk] = vv
    lhc_co_ref[kk] = vv

tw1 = lhc.b1.twiss4d()
tw1_ref = lhc_co_ref.b1.twiss4d()

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(tw1_ref.s, tw1_ref.x, label='Reference model')
plt.plot(tw1.s, tw1.x, label='Model with knobs')
plt.xlabel('s [m]')
plt.ylabel('x [m]')
plt.title('Horizontal closed orbit')
plt.legend()

plt.show()
