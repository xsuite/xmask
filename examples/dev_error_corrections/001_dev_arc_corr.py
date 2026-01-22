import xtrack as xt
import xdeps as xd

beam_name = 'b1'
env = xt.load('collider_00_from_mad.json')
line = env[f'lhc{beam_name}']

tw = env.lhcb1_co_ref.twiss4d() # Reference twiss

# Let's have a look at the b5 (k4)

tt = line.get_table(attr=True)

arc_name = '45'
start = f's.ds.r{arc_name[0]}.{beam_name}'
end = f'e.ds.r{arc_name[1]}.{beam_name}'
correction_knob = f'kcd.a45{beam_name}'

tt_range = tt.rows[start:end]