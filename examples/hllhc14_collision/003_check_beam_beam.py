import yaml
import json
import xobjects as xo
import xtrack as xt

with open('collider_02_bb_on.json', 'r') as fid:
    collider = xt.Multiline.from_dict(json.load(fid))

collider.build_trackers()

weak_beam = 'lhcb1'
strong_beam = 'lhcb2'

weak_line = collider[weak_beam]
strong_line = collider[strong_beam]

weak_tw = weak_line.twiss()
strong_tw = strong_line.twiss(reverse=True)


weak_tw = weak_line.twiss()
strong_tw = strong_line.twiss(reverse=True)



ip_name = 'ip1'

weak_survey = weak_line.survey(element0=ip_name)
strong_survey = strong_line.survey(element0=ip_name, reverse=True)
weak_bb = 'bb_lr.r1b1_24'
strong_bb = 'bb_lr.r1b2_24'

# collider.vars['beambeam_scale'] = 0
# collider.vars['on_disp'] = 0
# collider.vars['on_corr_co'] = 0

# check the orbit is not changed