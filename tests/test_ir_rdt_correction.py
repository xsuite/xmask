from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from xmask.lhc.correct_ir_rdts import (CIRCUIT, KEYWORD, MULTIPOLE, VALUE,
                                       apply_correction, calculate_correction,
                                       convert_line_to_madx_twiss)


def test_calculate_correction():
    """
    Unit test for calculate_correction function.
    The input IR is very simplified and only consists of two MQs and two 
    correctors. The beta-functions are all 1, so that the calculation 
    of the effective RDTs is straightforward.
    """

    # Mock Input
    @dataclass
    class MockElement:
        knl: List[float]
        ksl: List[float]
            

    class MockLine:
        def __init__(self):
            self.element_dict = {
                "mq.4l1": MockElement(knl=[0, 1], ksl=[0, 0]),
                "mcqx.3l1": MockElement(knl=[0, 0], ksl=[0, 0]),
                "mcqx.3r1": MockElement(knl=[0, 0], ksl=[0, 0]),
                "mq.4r1": MockElement(knl=[0, 1], ksl=[0, 0]),
            }

        def twiss(self):
            class MockTwiss:
                def to_pandas(self):
                    return pd.DataFrame({
                        "name": ["mq.4l1", "mcqx.3l1", "mcqx.3r1", "mq.4r1"],
                        "s": [1.0, 2.0, 3.0, 4.0],
                        "betx": [1., 1., 1., 1.],
                        "bety": [1., 1., 1., 1.]
                    })
            return MockTwiss()
    line = MockLine()

    # Run Correction
    correction = calculate_correction(line, regex_filter=r"M", **dict(ips=[1], rdts=["F2000"]))

    # Assert Results
    assert "kcqx3.l1" in correction[CIRCUIT].to_list()
    assert "kcqx3.r1" in correction[CIRCUIT].to_list()
    assert all(np.isclose(correction[VALUE], [-1., -1.]))  # to compensate for the [+1, +1] of the MQs at betax,y = 1.


def test_convert_line_to_madx_twiss():
    """
    Unit test for convert_line_to_madx_twiss function.

    Tests that the function correctly creates a MAD-X twiss TfsDataFrame from the data given in the line.
    Uses a mock line object and regex filter. Asserts that the function returns a DataFrame with expected
    index names, columns names, and values.
    """
    @dataclass
    class MockElement:
        knl: List[float]
        ksl: List[float]
            

    class MockLine:
        def __init__(self):
            self.element_dict = {
                "element1": MockElement(knl=[0.1, 0.2], ksl=[0.3, 0.4]),
                "element2": MockElement(knl=[0.3, 0.4], ksl=[0.1, 0.2]),
            }
            self.vars = {"circuit1": 0.0, "circuit2": 0.0}

        def twiss(self):
            class MockTwiss:
                def to_pandas(self):
                    return pd.DataFrame({
                        "name": ["element1", "element2"],
                        "s": [1.0, 2.0],
                        "betx": [1.1, 2.2],
                        "bety": [3.3, 4.4]
                    })
            return MockTwiss()

    # Run function with mock objects
    line = MockLine()
    df = convert_line_to_madx_twiss(line, regex_filter="element\d")

    # Assert that the returned DataFrame is as expected
    assert not df.empty
    assert df.index.to_list() == ["ELEMENT1", "ELEMENT2"]
    assert df.columns.to_list() == ["S", "BETX", "BETY", "K0L", "K1L",  "K0SL", "K1SL", KEYWORD]
    assert df.loc["ELEMENT1"].to_list() == [1.0, 1.1, 3.3, 0.1, 0.2, 0.3, 0.4, MULTIPOLE]
    assert df.loc["ELEMENT2"].to_list() == [2.0, 2.2, 4.4, 0.3, 0.4, 0.1, 0.2, MULTIPOLE]            
    
    
    df = convert_line_to_madx_twiss(line, regex_filter="ekdffs")
    assert df.empty


def test_apply_correction():
    """
    Unit test for apply_correction function.

    Tests that the function correctly applies the given correction to the given lines. 
    Uses a mock correction dataframe and lines object. Asserts that the function correctly 
    updates the variables of the given lines object based on the correction. 
    """
    # Mock correction dataframe
    correction = pd.DataFrame({
        "circuit": ["circuit1", "circuit2"],
        "value": [1.0, 2.0]
    })
    
    # Mock Line object
    class MockLine:
        def __init__(self):
            self.vars = {"circuit1": 0.0, "circuit2": 0.0}


    lines = [MockLine(), MockLine()]

    # Apply correction
    apply_correction(*lines, correction=correction)

    # Assert that variables were updated correctly
    assert lines[0].vars["circuit1"] == 1.0
    assert lines[0].vars["circuit2"] == 2.0
    assert lines[1].vars["circuit1"] == 1.0
    assert lines[1].vars["circuit2"] == 2.0
