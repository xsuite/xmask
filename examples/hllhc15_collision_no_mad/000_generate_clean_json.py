from cpymad.madx import Madx
import xtrack as xt

mad = Madx()
mad.input(
    '''
    ! Get the toolkit
    call,file="macro.madx";
    call,file="lhc.seq";

    !Install HL-LHC
    call, file="hllhc_sequence.madx";

    ! Install crab cavities (they are off)
    call, file='enable_crabcavities.madx';
    on_crab1 = 0;
    on_crab5 = 0;

    beam, sequence=lhcb1, particle=proton, pc=7000;
    beam, sequence=lhcb2, particle=proton, pc=7000, bv=-1;

    set, format="12d", "-18.12e", "25s";
    save, file="temp_lhc_thick.seq";

    ''')

lhc = xt.load('temp_lhc_thick.seq', s_tol=1e-6,
              _rbend_correct_k0=True, # LHC sequences are defined with rbarc=False
              reverse_lines=['lhcb2'])

lhc.lines['b1'] = lhc.lhcb1
lhc.lines['b2'] = lhc.lhcb2
del lhc.lines['lhcb1']
del lhc.lines['lhcb2']

for ll in ['b1', 'b2']:
    lhc.lines[ll].slice_thick_elements(
        slicing_strategies=[
            xt.Strategy(slicing=xt.Teapot(1)),
            xt.Strategy(element_type=xt.UniformSolenoid, slicing=None),
            xt.Strategy(name=r'mb\..*',    slicing=xt.Teapot(2)),
            xt.Strategy(name=r'mq\..*',    slicing=xt.Teapot(2)),
            xt.Strategy(name=r'mqxa\..*',  slicing=xt.Teapot(16)),  # old triplet
            xt.Strategy(name=r'mqxb\..*',  slicing=xt.Teapot(16)),  # old triplet
            xt.Strategy(name=r'mqxc\..*',  slicing=xt.Teapot(16)),  # new mqxa (q1,q3)
            xt.Strategy(name=r'mqxd\..*',  slicing=xt.Teapot(16)),  # new mqxb (q2a,q2b)
            xt.Strategy(name=r'mqxfa\..*', slicing=xt.Teapot(16)),  # new (q1,q3 v1.1)
            xt.Strategy(name=r'mqxfb\..*', slicing=xt.Teapot(16)),  # new (q2a,q2b v1.1)
            xt.Strategy(name=r'mbxa\..*',  slicing=xt.Teapot(4)),   # new d1
            xt.Strategy(name=r'mbxf\..*',  slicing=xt.Teapot(4)),   # new d1 (v1.1)
            xt.Strategy(name=r'mbrd\..*',  slicing=xt.Teapot(4)),   # new d2 (if needed)
            xt.Strategy(name=r'mqyy\..*',  slicing=xt.Teapot(4)),   # new q4
            xt.Strategy(name=r'mqyl\..*',  slicing=xt.Teapot(4)),   # new q5
            xt.Strategy(name=r'mbx\..*',    slicing=xt.Teapot(4)),
            xt.Strategy(name=r'mbrb\..*',   slicing=xt.Teapot(4)),
            xt.Strategy(name=r'mbrc\..*',   slicing=xt.Teapot(4)),
            xt.Strategy(name=r'mbrs\..*',   slicing=xt.Teapot(4)),
            xt.Strategy(name=r'mbh\..*',    slicing=xt.Teapot(4)),
            xt.Strategy(name=r'mqwa\..*',   slicing=xt.Teapot(4)),
            xt.Strategy(name=r'mqwb\..*',   slicing=xt.Teapot(4)),
            xt.Strategy(name=r'mqy\..*',    slicing=xt.Teapot(4)),
            xt.Strategy(name=r'mqm\..*',    slicing=xt.Teapot(4)),
            xt.Strategy(name=r'mqmc\..*',   slicing=xt.Teapot(4)),
            xt.Strategy(name=r'mqml\..*',   slicing=xt.Teapot(4)),
            xt.Strategy(name=r'mqtlh\..*',  slicing=xt.Teapot(2)),
            xt.Strategy(name=r'mqtli\..*',  slicing=xt.Teapot(2)),
            xt.Strategy(name=r'mqt\..*',    slicing=xt.Teapot(2)),
])

lhc.vars.load('opt_round_150_1500_thin.madx')

prrrrr

########################
# Match coupling knobs #
########################

lhc.b1.set_particle_ref('proton', energy0=7000e9)
lhc.b2.set_particle_ref('proton', energy0=7000e9)

# Select circuits with appropriate weights
vary = {}
vary['b1']=[xt.VaryList(['kqs.a23b1', 'kqs.a67b1'], step=5e-5),
      xt.VaryList(['kqs.l4b1', 'kqs.l8b1','kqs.r3b1', 'kqs.r7b1'],
                  weight=2, step=5e-5)]

vary['b2']=[xt.VaryList(['kqs.a34b2', 'kqs.a78b2'], step=5e-5),
      xt.VaryList(['kqs.l3b2', 'kqs.r4b2','kqs.r6b2', 'kqs.l7b2','kqs.r8b2'],
                  weight=2, step=5e-5)]

for bb in ['b1', 'b2']:
    c_min_match = 1e-4
    opt_re = lhc[bb].match_knob(knob_name=f'c_minus_re.{bb}',
        knob_value_start=0, knob_value_end=c_min_match,
        run=False, method='4d',
        vary=vary[bb],
        targets=[
            xt.Target('c_minus_re_0', value=c_min_match, tol=1e-8),
            xt.Target('c_minus_im_0', value=0,           tol=1e-8),
        ])
    opt_re.solve()
    opt_re.generate_knob()

    # Match c_minus_im.b1
    opt_im = lhc[bb].match_knob(knob_name=f'c_minus_im.{bb}',
        knob_value_start=0, knob_value_end=c_min_match,
        run=False, method='4d',
        vary=vary[bb],
        targets=[
            xt.Target('c_minus_re_0', value=0,           tol=1e-8),
            xt.Target('c_minus_im_0', value=c_min_match, tol=1e-8),
        ])
    opt_im.solve()
    opt_im.generate_knob()

# Test the knobs
lhc.b1['c_minus_re.b1'] = 1e-3
lhc.b1['c_minus_im.b1'] = 0
tw = lhc.b1.twiss4d()
print('Test c_minus_re.b1 = 1e-3')
print('c_minus_re_0 = ', tw.c_minus_re_0) # is 0.00099998
print('c_minus_im_0 = ', tw.c_minus_im_0) # is 4.05e-09

lhc.b1['c_minus_re.b1'] = 0
lhc.b1['c_minus_im.b1'] = 1e-3
tw = lhc.b1.twiss4d()
print('Test c_minus_im.b1 = 1e-3')
print('c_minus_re_0 = ', tw.c_minus_re_0) # is 2.16e-8
print('c_minus_im_0 = ', tw.c_minus_im_0) # is 0.001000003

lhc.b2['c_minus_re.b2'] = 1e-3
lhc.b2['c_minus_im.b2'] = 0
tw = lhc.b2.twiss4d()
print('Test c_minus_re.b2 = 1e-3')
print('c_minus_re_0 = ', tw.c_minus_re_0) # is
print('c_minus_im_0 = ', tw.c_minus_im_0) # is 4.05e-09

lhc.b2['c_minus_re.b2'] = 0
lhc.b2['c_minus_im.b2'] = 1e-3
tw = lhc.b2.twiss4d()
print('Test c_minus_im.b2 = 1e-3')
print('c_minus_re_0 = ', tw.c_minus_re_0) # is 2.16e-8
print('c_minus_im_0 = ', tw.c_minus_im_0) # is 0.001000003

# All to zero
lhc.b1['c_minus_re.b1'] = 0
lhc.b1['c_minus_im.b1'] = 0
lhc.b2['c_minus_re.b2'] = 0
lhc.b2['c_minus_im.b2'] = 0

lhc.b1.particle_ref = None
lhc.b2.particle_ref = None
lhc.to_json('lhc.json')
