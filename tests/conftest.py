"""
Additional tools for pytests.
The name ``conftest.py`` is chosen as it is used by pytest.
Fixtures defined in here are discovered by all tests automatically.

See also https://stackoverflow.com/a/34520971 .
"""
from pathlib import Path

import xtrack as xt
from _complementary_hllhc14 import apply_optics, build_sequence
from cpymad.madx import Madx
from pytest import fixture

import xmask as xm
import xmask.lhc as xmlhc


test_data_dir = Path(__file__).parent.parent / "test_data"

# Fixtures ---------------------------------------------------------------------
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
        collider = _create_hllhc14_b1(tmp_path, enable_coupling_knobs=True)
        collider.to_json(json_path)
        return collider
    
    return xt.Multiline.from_json(json_path)


@fixture(scope="function")
def hllhc14_beam1_no_coupling_knobs(tmp_path_factory):
    """ Create hllhc14 beam 1, without coupling knobs, for testing. 
    The scope of this fixture is "function" so that each time the collider
    is loaded anew. But tmp_path_factory scope is "session" so that if 
    it is saved once to a .json it is simply loaded. """
    tmp_path = tmp_path_factory.mktemp("hllhc14_b1_no_coupling_knobs_fixture")
    # tmp_path = Path()  # for debugging
    json_path = tmp_path / 'collider_b1.json'

    if not json_path.is_file():
        collider = _create_hllhc14_b1(tmp_path, enable_coupling_knobs=False)
        collider.to_json(json_path)
        return collider
    
    return xt.Multiline.from_json(json_path)


# Helper -----------------------------------------------------------------------
def _create_hllhc14_b1(output_dir: Path, enable_coupling_knobs: bool = True):
    # Make mad environment
    xm.make_mad_environment(links={
        'acc-models-lhc': str(test_data_dir / 'hllhc14')})

    # Start mad
    mad_b1 = Madx(command_log=str(output_dir / "mad_collider.log"))

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
        enable_knob_synthesis=enable_coupling_knobs,
        rename_coupling_knobs=False,
        pars_for_imperfections={},
        ver_lhc_run=None,
        ver_hllhc_optics=1.4)

    assert len(collider.lines.keys()) == 2  # b1 and b1_co_ref
    return collider
