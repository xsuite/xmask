# %%
import yaml
import json
import xobjects as xo
import xtrack as xt
import xfields as xf

from matplotlib import pyplot as plt

    # %%
with open('collider_02_bb_on.json', 'r') as fid:
    collider = xt.Multiline.from_dict(json.load(fid))

collider.build_trackers()

weak_beam = 'lhcb1'
strong_beam = 'lhcb2'
ip_name = 'ip1'

weak_line = collider[weak_beam]
strong_line = collider[strong_beam]

weak_tw = weak_line.twiss()
strong_tw = strong_line.twiss(reverse=True)

weak_survey = weak_line.survey(element0=ip_name)
strong_survey = strong_line.survey(element0=ip_name, reverse=True)

weak_bb = 'bb_lr.r1b1_24'
strong_bb = 'bb_lr.r1b2_24'

# %% test that the closed orbit is not affected by the BB lenses

import numpy as np

collider.vars['beambeam_scale'] = 1

# Q: if I change the on_disp/on_corr_co knobs the checks of the BB closed orbit invariance failed.
# I think this is normal (due to the fact that the BB dipole kick is not updated if the orbit is updated).
 
collider.vars['on_disp'] = 1  
collider.vars['on_corr_co'] = 1


weak_tw = weak_line.twiss()
weak_co_bb_on = {xx:weak_tw[xx] for xx in ['x', 'px', 'y', 'py', 'zeta', 'delta']}

collider.vars['beambeam_scale'] = 0
weak_tw = weak_line.twiss()
weak_co_bb_off = {xx:weak_tw[xx] for xx in ['x', 'px', 'y', 'py', 'zeta', 'delta']}

collider.vars['beambeam_scale'] = 1

# Q: why delta is not a 'numpy.ndarray' but a 'xobjects.context_cpu.LinkedArrayCpu'?
# is there a missed casting?

for ii in ['x', 'px', 'y', 'py', 'zeta', 'delta']:
    print(f'Type of {ii} is {type(weak_co_bb_on[ii])}')

# %% there are other elements that are 'xobjects.context_cpu.LinkedArrayCpu' 
for ii in weak_tw.keys():
    if type(weak_tw[ii]) == xo.context_cpu.LinkedArrayCpu:
        print(f'Type of {ii} is {type(weak_tw[ii])}')

# %% Test

for ii in ['x', 'px', 'y', 'py', 'zeta', 'delta']:
    np.testing.assert_allclose(weak_co_bb_on[ii], weak_co_bb_off[ii],  atol=1.2e-10, rtol=0)
    print(f'{ii} is OK')

# %%
# Q: is there something fishy with the zeta?
for ii in ['x', 'px', 'y', 'py', 'zeta', 'delta']:
    plt.plot(weak_tw['s'], weak_co_bb_on[ii]-weak_co_bb_off[ii])
    plt.xlabel('s [m]')
    plt.ylabel(ii)
    plt.show()

# %% Check of the number of BB lenses
list_of_BeamBeamBiGaussian2D = []
list_of_BeamBeamBiGaussian2D_name = []
list_of_BeamBeamBiGaussian3D = []
list_of_BeamBeamBiGaussian3D_name = []


for ii in weak_line.elements:
    if type(ii)== xf.beam_elements.beambeam2d.BeamBeamBiGaussian2D:
        list_of_BeamBeamBiGaussian2D.append(ii)
        list_of_BeamBeamBiGaussian2D_name.append(weak_line.element_names[ii])
    if type(ii)== xf.beam_elements.beambeam3d.BeamBeamBiGaussian3D:
        list_of_BeamBeamBiGaussian3D.append(ii)
    


# collider.vars['beambeam_scale'] = 0
# collider.vars['on_disp'] = 0
# collider.vars['on_corr_co'] = 0

# check the orbit is not changed
# %%
dict_of_BeamBeamBiGaussian2D = {}
for ii in weak_line.element_dict:
    if type(weak_line.element_dict[ii])== xf.beam_elements.beambeam2d.BeamBeamBiGaussian2D:
        dict_of_BeamBeamBiGaussian2D[ii]=weak_line.element_dict[ii]


# %%
