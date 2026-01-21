import xtrack as xt
import xdeps as xd

env = xt.load('collider_00_from_mad.json')

tw = env.lhcb1_co_ref.twiss4d() # Reference twiss

# Let's try to compute the IP5 b4 correctors (kcsx3.l5 and kcsx3.r5)
line = env.lhcb1
start = 'dfxj.4l5'
end = 'dfxj.4r5'
correction_knobs = ['kcox3.l5', 'kcox3.r5']
multipole = 'k3l'
rdt_indeces = [(0, 4), (4, 0)]

# for kk in correction_knobs:
#     env[kk] = 0.0

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

ref_names = [nn.replace('/lhcb1', '') for nn in tt_integral.env_name]
tw_integral = tw.rows[ref_names]
rdt_terms = {}
for rdt_i in rdt_indeces:
    assert len(rdt_i) == 2
    ii = rdt_i[0]
    jj = rdt_i[1]

    r_ii_jj = (tw_integral.betx ** (ii / 2) * tw_integral.bety ** (jj / 2)
               * tt_integral[multipole])
    rdt_terms[rdt_i] = r_ii_jj.sum()

