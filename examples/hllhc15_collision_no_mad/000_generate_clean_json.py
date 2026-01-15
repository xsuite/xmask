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

lhc.vars.load('opt_round_150_1500.madx')

########################
# Match coupling knobs #
########################

lhc.b1.set_particle_ref('proton', energy0=7000e9)
lhc.b2.set_particle_ref('proton', energy0=7000e9)

# Select circuits with appropriate weights
vary=[xt.VaryList(['kqs.a23b1', 'kqs.a67b1'], step=5e-5),
      xt.VaryList(['kqs.l4b1', 'kqs.l8b1','kqs.r3b1', 'kqs.r7b1'],
                  weight=2, step=5e-5)]

# Match c_minus_re.b1
c_min_match = 1e-4
opt_re = lhc.b1.match_knob(knob_name='c_minus_re.b1',
    knob_value_start=0, knob_value_end=c_min_match,
    run=False, method='4d',
    vary=vary,
    targets=[
        xt.Target('c_minus_re_0', value=c_min_match, tol=1e-8),
        xt.Target('c_minus_im_0', value=0,           tol=1e-8),
    ])
opt_re.solve()
opt_re.generate_knob()

# Match c_minus_im.b1
opt_im = lhc.b1.match_knob(knob_name='c_minus_im.b1',
    knob_value_start=0, knob_value_end=c_min_match,
    run=False, method='4d',
    vary=vary,
    targets=[
        xt.Target('c_minus_re_0', value=0,           tol=1e-8),
        xt.Target('c_minus_im_0', value=c_min_match, tol=1e-8),
    ])
opt_im.solve()
opt_im.generate_knob()

# Test the knob
lhc.b1['c_minus_re.b1'] = 1e-3
tw = lhc.b1.twiss4d()
tw.c_minus_re_0 # is 0.00099998
tw.c_minus_im_0 # is 4.05e-09

lhc.b1['c_minus_re.b1'] = 0
lhc.b1['c_minus_im.b1'] = 1e-3
tw = lhc.b1.twiss4d()
tw.c_minus_re_0 # is 2.16e-8
tw.c_minus_im_0 # is 0.001000003

lhc.to_json('lhc.json')
