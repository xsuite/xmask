import xtrack as xt

env = xt.load('collider_00_from_mad.json')

tw = env.lhcb1_co_ref.twiss4d() # Reference twiss

# Let's try to compute the IP5 b4 correctors (kcsx3.l5 and kcsx3.r5)
line = env.lhcb1
start = 'dfxj.4l5'
end = 'dfxj.4r5'

tt = line.get_table(attr=True)

tt_range = tt.rows[start:end]
