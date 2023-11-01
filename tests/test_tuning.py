import xtrack as xt

from xmask.tuning import analytical_coupling_correction


def test_analytical_coupling_correction(hllhc14_beam1):
    lhcb1: xt. Line = hllhc14_beam1.lines['lhcb1']
    lhcb1.twiss_default["method"] = "4d"
    reference = lhcb1.twiss().c_minus

    # add some coupling
    lhcb1.vars["kqsx3.r5"] = 1e-5
    before_corr = lhcb1.twiss().c_minus
    
    # perform correction
    analytical_coupling_correction(
        line=lhcb1, 
        knob_names={
            "c_minus_knob_1": "c_minus_re_b1", 
            "c_minus_knob_2": "c_minus_im_b1"
            },
    ) 
    after_corr = lhcb1.twiss().c_minus

    # Test correction
    # print("reference", reference)
    # print("before_corr", before_corr)
    # print("after_corr", after_corr)
    assert lhcb1.vars["c_minus_re_b1"] != 0
    assert lhcb1.vars["c_minus_im_b1"] != 0
    assert after_corr < before_corr

