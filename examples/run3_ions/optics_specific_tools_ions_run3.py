import warnings

def build_sequence(mad, mylhcbeam, **kwargs):

    # Select beam

    #slicefactor = 2 # For testing
    slicefactor = 8 # For production

    mylhcbeam = int(mylhcbeam)

    mad.input('ver_lhc_run = 3')

    mad.input(f'mylhcbeam = {mylhcbeam}')
    mad.input('option, -echo,warn, -info;')

    # optics dependent macros (for splitting)
    mad.call('optics_runII/2018/toolkit/macro.madx')

    # # Redefine macros
    # _redefine_crossing_save_disable_restore(mad)

    # # optics independent macros
    # mad.call('tools/optics_indep_macros.madx')

    assert mylhcbeam in [1, 2, 4], "Invalid mylhcbeam (it should be in [1, 2, 4])"

    if mylhcbeam in [1, 2]:
        mad.call('optics_runII/2018/lhc_as-built.seq')
    else:
        mad.call('optics_runII/2018/lhcb4_as-built.seq')

    # New IR7 MQW layout and cabling
    mad.call('optics_runIII/RunIII_dev/IR7-Run3seqedit.madx')

    # Makethin part
    if slicefactor > 0:
        # the variable in the macro is slicefactor
        mad.input(f'slicefactor={slicefactor};')
        mad.call('optics_runII/2018/toolkit/myslice.madx')
        mad.beam()
        for my_sequence in ['lhcb1','lhcb2']:
            if my_sequence in list(mad.sequence):
                mad.input(f'use, sequence={my_sequence}; makethin,'
                     f'sequence={my_sequence}, style=teapot, makedipedge=true;')
    else:
        warnings.warn('The sequences are not thin!')

    # Cycling w.r.t. to IP3 (mandatory to find closed orbit in collision in the presence of errors)
    for my_sequence in ['lhcb1','lhcb2']:
        if my_sequence in list(mad.sequence):
            mad.input(f'seqedit, sequence={my_sequence}; flatten;'
                        'cycle, start=IP3; flatten; endedit;')

def apply_optics(mad, optics_file):
    mad.call(optics_file)
    mad.call('optics_runIII/ir7_strengths.madx')
    mad.input('on_alice := on_alice_normalized * 7000./nrj;')
    mad.input('on_lhcb := on_lhcb_normalized * 7000./nrj;')
