import xtrack as xt

def slice_lattice(line):
    line.slice_thick_elements(slicing_strategies=[
        xt.Strategy(None), # By default leave untouched
        xt.Strategy(element_type=xt.Cavity,     slicing=xt.Teapot(1)),
        xt.Strategy(element_type=xt.CrabCavity, slicing=xt.Teapot(1)),
        xt.Strategy(element_type=xt.Sextupole,  slicing=xt.Teapot(1)),
        xt.Strategy(element_type=xt.Octupole,   slicing=xt.Teapot(1)),
        xt.Strategy(element_type=xt.Multipole,  slicing=xt.Teapot(1)), # in case correctors are thick multipoles
        xt.Strategy(name=r'mb\..*',    slicing=xt.Teapot(2)),
        xt.Strategy(name=r'mq\..*',    slicing=xt.Teapot(4)), # !!!!! was 2!!!!!!
        xt.Strategy(name=r'mqxa\..*',  slicing=xt.Teapot(16)),  #old triplet
        xt.Strategy(name=r'mqxb\..*',  slicing=xt.Teapot(16)),  #old triplet
        xt.Strategy(name=r'mqxc\..*',  slicing=xt.Teapot(16)),  #new mqxa (q1,q3)
        xt.Strategy(name=r'mqxd\..*',  slicing=xt.Teapot(16)),  #new mqxb (q2a,q2b)
        xt.Strategy(name=r'mqxfa\..*', slicing=xt.Teapot(16)),  #new (q1,q3 v1.1)
        xt.Strategy(name=r'mqxfb\..*', slicing=xt.Teapot(16)),  #new (q2a,q2b v1.1)
        xt.Strategy(name=r'mbxa\..*',  slicing=xt.Teapot(4)),  #new d1
        xt.Strategy(name=r'mbxf\..*',  slicing=xt.Teapot(4)),  #new d1 (v1.1)
        xt.Strategy(name=r'mbrd\..*',  slicing=xt.Teapot(4)),  #new d2 (if needed)
        xt.Strategy(name=r'mqyy\..*',  slicing=xt.Teapot(4)),  #new q4
        xt.Strategy(name=r'mqyl\..*',  slicing=xt.Teapot(4)),  #new q5
        xt.Strategy(name=r'mbw\..*',   slicing=xt.Teapot(4)),
        xt.Strategy(name=r'mbx\..*',   slicing=xt.Teapot(4)),
        xt.Strategy(name=r'mbrb\..*',  slicing=xt.Teapot(4)),
        xt.Strategy(name=r'mbrc\..*',  slicing=xt.Teapot(4)),
        xt.Strategy(name=r'mbrs\..*',  slicing=xt.Teapot(4)),
        xt.Strategy(name=r'mqwa\..*',  slicing=xt.Teapot(4)),
        xt.Strategy(name=r'mqwb\..*',  slicing=xt.Teapot(4)),
        xt.Strategy(name=r'mqy\..*',   slicing=xt.Teapot(4)),
        xt.Strategy(name=r'mqm\..*',   slicing=xt.Teapot(4)),
        xt.Strategy(name=r'mqmc\..*',  slicing=xt.Teapot(4)),
        xt.Strategy(name=r'mqml\..*',  slicing=xt.Teapot(4)),
        xt.Strategy(name=r'mqtlh\..*', slicing=xt.Teapot(2)),
        xt.Strategy(name=r'mqtli\..*', slicing=xt.Teapot(2)),
        xt.Strategy(name=r'mqt\..*',   slicing=xt.Teapot(2)),
        xt.Strategy(name=r'mqs\..*',   slicing=xt.Teapot(1)),
    ])