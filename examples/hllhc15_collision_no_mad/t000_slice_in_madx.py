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

    ! Slice nominal sequence
    exec, myslice;

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

lhc.vars.load('opt_round_150_1500_thin.madx')

lhc.b1.set_particle_ref('proton', energy0=7000e9)
lhc.b2.set_particle_ref('proton', energy0=7000e9)

tw1 = lhc.b1.twiss4d()

lhc.to_json('lhc_slice_mad.json')