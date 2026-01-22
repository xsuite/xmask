import xtrack as xt
import xdeps as xd

beam_name = 'b1'
env = xt.load('collider_00_from_mad_with_errors.json')
line = env[f'lhc{beam_name}']

# Reference twiss
env_no_err = xt.load('collider_00_from_mad_no_errors.json')
tw = env_no_err[f'lhc{beam_name}'].twiss4d() # Reference twiss

# Let's have a look at the b5 (k4)

arc_name = '45'
start = f's.ds.r{arc_name[0]}.{beam_name}'
end = f'e.ds.r{arc_name[1]}.{beam_name}'
correction_knobs = [f'kcd.a45{beam_name}']
multipole = 'k4l'

for kk in correction_knobs:
    env[kk] = 0.0

tt = line.get_table(attr=True)
tt_range = tt.rows[start:end]

# Identify elements controlled by correction knobs
elements_in_range = set(tt_range.env_name)
correction_elements = []
for kk in correction_knobs:
    for tt in line.ref[kk]._find_dependant_targets():
        if isinstance(tt, xd.refs.ItemRef) and tt._key in elements_in_range:
            correction_elements.append(tt._key)

mask_corr = tt_range.rows.mask[list(correction_elements)]
tt_integral = tt_range.rows[(tt_range[multipole] != 0) | (mask_corr)]
tw_integral = tw.rows[tt_integral.env_name]