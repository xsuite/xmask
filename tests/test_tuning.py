from pathlib import Path
from pytest import fixture
import xtrack as xt
import xmask as xm
import xmask.lhc as xmlhc
from cpymad.madx import Madx
from _complementary_hllhc14 import build_sequence, apply_optics
from xmask.tuning import analytical_coupling_correction


test_data_dir = Path(__file__).parent.parent / "test_data"

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


@fixture(scope="function")
def hllhc14_beam1(tmp_path_factory):
    """ Create hllhc14 beam 1 for testing. 
    The scope of this fixture is "function" so that each time the collider
    is loaded anew. But tmp_path_factory scope is "session" so that if 
    it is saved once to a .json it is simply loaded. """
    tmp_path = tmp_path_factory.mktemp("hllhc14_b1_fixture")
    tmp_path = Path()  # for debugging
    json_path = tmp_path / 'collider_b1.json'

    if not json_path.is_file():
        # Make mad environment
        xm.make_mad_environment(links={
            'acc-models-lhc': str(test_data_dir / 'hllhc14')})

        # Start mad
        mad_b1 = Madx(command_log="mad_collider.log")

        # Build sequence
        build_sequence(mad_b1, mylhcbeam=1)

        # Apply optics
        apply_optics(mad_b1,
            optics_file="acc-models-lhc/round/opt_round_150_1500_thin.madx")

        # Build xsuite collider
        collider = xmlhc.build_xsuite_collider(
            sequence_b1=mad_b1.sequence.lhcb1,
            sequence_b2=None,
            sequence_b4=None,
            beam_config={'lhcb1':{'beam_energy_tot': 7000}},
            enable_imperfections=False,
            enable_knob_synthesis='_mock_for_testing',
            rename_coupling_knobs=True,
            pars_for_imperfections={},
            ver_lhc_run=None,
            ver_hllhc_optics=1.4)

        assert len(collider.lines.keys()) == 2  # b1 and b1_co_ref
        collider.to_json(json_path)
        return collider
    
    return xt.Multiline.from_json(json_path)