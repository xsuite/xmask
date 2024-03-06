from dataclasses import dataclass, fields
from typing import List, Optional

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
    line = MockLine(
        MyElement(name="mq.4l1",   s=1, knl=[0, 1], ksl=[0, 0]),
        MyElement(name="mcqx.3l1", s=2, knl=[0, 0], ksl=[0, 0]),
        MyElement(name="mcqx.3r1", s=3, knl=[0, 0], ksl=[0, 0]),
        MyElement(name="mq.4r1",   s=4, knl=[0, 1], ksl=[0, 0]),
    )

    # Run Correction
    correction = calculate_correction(line, beams=[1], regex_filter=r"M", **dict(ips=[1], rdts=["F2000"]))

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
    line = MockLine(
        MyElement(name="element1", s=1.0, betx=1.1, bety=3.3, knl=[0.1, 0.2], ksl=[0.3, 0.4]),
        MyElement(name="element2", s=2.0, betx=2.2, bety=4.4, knl=[0.3, 0.4], ksl=[0.1, 0.2]),
    )
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
        "name": ["magnet1", "magnet2"],
        "circuit": ["circuit1", "circuit2"],
        "value": [1.0, 2.0]
    })

    # Mock Element and Line (very few attributes needed)
    @dataclass
    class MockElement:
        length: float
    
    class MockLine:
        def __init__(self, length):
            self.vars = {"circuit1": 0.0, "circuit2": 0.0}
            self.element_dict = {
                "magnet1": MockElement(length=length),
                "magnet2": MockElement(length=length)
            }

    lines = [
        MockLine(length=1.0), 
        MockLine(length=2.0)
    ]

    # Apply correction
    apply_correction(*lines, correction=correction)

    correction = correction.set_index("name")

    for line in lines:
        for idx in (1, 2):
            assert line.vars[f"circuit{idx}"] == (
                correction.loc[f"magnet{idx}", "value"] / line.element_dict[f"magnet{idx}"].length
            )


# Some Classes to Mock XLine behaviour for the tests -------------------------------------------------------------------
@dataclass
class MyElement:
    """ Mocks a line element in a sense that it has knl and ksl attributes, 
    but is also used here to define the entries of an element in the element table/twiss pandas DataFrame."""
    name: str
    s: float
    knl: List[float]
    ksl: List[float]
    betx: Optional[float] = 1
    bety: Optional[float] = 1
    element_type: Optional[str] = "multipole"

    def to_data_dict(self):
        d = {f.name: getattr(self, f.name) for f in fields(self) if f.name not in ["knl", "ksl"]}
        d.update({f"k{idx}l": value for idx, value in enumerate(self.knl)})
        d.update({f"k{idx}sl": value for idx, value in enumerate(self.ksl)})
        return d


@dataclass
class MockElement:
    """ Mocks a line element in a sense that it has knl and ksl attributes."""
    knl: List[float]
    ksl: List[float]


class MockLine:
    """ Creates a line-object that has the needed attributes of line as used in the tests."""
    def __init__(self, *elements: MyElement):
        self.element_dict = {element.name: MockElement(element.knl, element.ksl) for element in elements}
        self._pandas = pd.DataFrame(
                    index=self.element_dict.keys(),
                    data=[element.to_data_dict() for element in elements],
                )

    def twiss(outer_self):
        class MockTwiss:
            def to_pandas(self):
                return outer_self._pandas.copy().drop("element_type", axis=1)
        return MockTwiss()
    
    def get_table(outer_self):
        class MockTable:
            def __getitem__(self, key):
                df = outer_self._pandas.copy().drop(["betx", "bety"], axis=1)
                return df.loc[key[1], key[0]]
        return MockTable()
