
import numpy as np
import xtrack as xt
from integral_correction import IntegralCorrection

beam_name = 'b1'
env = xt.load('collider_00_from_mad_with_errors.json')
line = env[f'lhc{beam_name}']

# Reference twiss
env_no_err = xt.load('collider_00_from_mad_no_errors.json')
tw = env_no_err[f'lhc{beam_name}'].twiss4d() # Reference twiss

# Let's have a look at the b5 (k4) # decapole

arc_name = '45'
start = f's.ds.r{arc_name[0]}.{beam_name}'
end = f'e.ds.l{arc_name[1]}.{beam_name}'
correction_knobs = [f'kcd.a45{beam_name}']
multipole = 'k4l'
target_quantities={'k4l': lambda tw, tt: tt[multipole].sum()}

tt = line.get_table()
scale_multipole = np.zeros_like(tt.s)
scale_multipole[tt.rows.mask[r'mb.*']] = 1.0 # only bends as sources
scale_multipole[tt.rows.mask[r'mc.*']] = 1.0 # all magnets called mcXXX used as correctors

# Usage:
rdt_contrib = IntegralCorrection(
                         line=env['lhcb1'],
                         tw=tw,
                         start=start,
                         end=end,
                         correction_knobs=correction_knobs,
                         multipole=multipole,
                         ip=None,
                         target_quantities=target_quantities,
                         generated_knob_name='on_corr_k3_ip5',
                         scale_multipole=scale_multipole)
print("Original correction:")
rdt_contrib.print_corrections()

rdt_contrib.clear_corrections()
opt = rdt_contrib.correct()

print("Before setting the knob:")
rdt_contrib.print_corrections()

env[opt.knob_name] = 1.0
print("After setting the knob:")
rdt_contrib.print_corrections()

# for kk in correction_knobs:
#     env[kk] = 0.0

# tt0 = line.get_table(attr=True)
# tt0_range = tt0.rows[start:end]

# # Identify elements controlled by correction knobs
# elements_in_range = set(tt0_range.env_name)
# correction_elements = []
# for kk in correction_knobs:
#     for ttar in line.ref[kk]._find_dependant_targets():
#         if isinstance(ttar, xd.refs.ItemRef) and ttar._key in elements_in_range:
#             correction_elements.append(ttar._key)

# mask_corr = tt0_range.rows.mask[list(correction_elements)]
# tt_integral = tt0_range.rows[(tt0_range[multipole] != 0) | (mask_corr)]
# tw_integral = tw.rows[tt_integral.name]



# env[correction_knobs[0]] = 0