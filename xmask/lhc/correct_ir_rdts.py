
"""
IRNL RDT Correction
-------------------

In this module, wrapper around the irnl-correction functions are 
provided which allow running the script with XTrack lines.
In this correction, the powering for the correctors in the IRs
are calculated by minimizing the local RDTs.

This correction implicitly assumes, that each corrector magnet
is powered by a single knob and that the KNL value of the magnet 
is calculated by knob-value * length-of-magnet.

See: https://github.com/pylhc/irnl_rdt_correction/blob/master/latex/note.pdf
"""
from typing import Dict
from xtrack.line import Line
from irnl_rdt_correction.main import irnl_rdt_correction
from irnl_rdt_correction.constants import KEYWORD, MULTIPOLE
import re
from pandas import DataFrame

DEFAULT_IR_FILTER: str = "M(QS?X|BX|BRC|C[SODT]S?X)"

# Corrector DataFrame Columns
NAME: str = "name"
CIRCUIT: str = "circuit"
VALUE: str = "value"

# XTrack/MADX order notation
XTRACK_TO_MADX: Dict[str, str] = {
    "knl": "K{}L",
    "ksl": "K{}SL",
}

def calculate_correction(*lines: Line, regex_filter: str = DEFAULT_IR_FILTER, **kwargs) -> DataFrame:
    """Run the correction with the given Line instances from XTrack.
    A combined correction including the optics of all given lines is performed.

    Args:
        lines (Line): XTrack line objects to run the correction on. 
        regex_filter (str): Regular expression to filter the elements, 
                            which contribute to the RDTs.

    Keyword Args:
        See :meth:`irnl_rdt_correction.main.irnl_rdt_correction`
    """
    _, correction_df = irnl_rdt_correction(
            beams=kwargs.pop("beams", list(range(1, len(lines) + 1))),
            twiss=[convert_line_to_madx_twiss(line, regex_filter) for line in lines],
            errors=None,
            ignore_missing_columns=True,  # required, as errors are not given explicitly
            **kwargs,
    )

    for col in (NAME, CIRCUIT):
        correction_df[col] = correction_df[col].str.lower()  # convert back to xtrack style
    return correction_df


def convert_line_to_madx_twiss(line: Line, regex_filter: str = None) -> DataFrame:
    """Create a MAD-X twiss TfsDataFrame from the data given in line.

    Args:
        line (Line): XTrack line object to convert.

    Returns:
        DataFrame: More MAD-X twiss-like DataFrame.
    """
    twiss = line.twiss()
    twiss_df = twiss.to_pandas()
    twiss_df.columns = twiss_df.columns.str.upper()  # MAD-X style 
    twiss_df = twiss_df.set_index(NAME.upper(), drop=True)
    if regex_filter is not None:
        twiss_df = twiss_df.loc[twiss_df.index.str.match(regex_filter, flags=re.IGNORECASE), :]

    for element_name in twiss_df.index:
        element = line.element_dict[element_name]
        for kl_name, kl_madx in XTRACK_TO_MADX.items():
            try:
                kl_list = getattr(element, kl_name)
            except AttributeError:
                continue

            columns = [kl_madx.format(order) for order in range(len(kl_list))]
            twiss_df.loc[element_name, columns] = kl_list

    twiss_df = twiss_df.fillna(0)
    twiss_df[KEYWORD] = line.get_table()["element_type", twiss_df.index.to_list()]
    twiss_df[KEYWORD] = twiss_df[KEYWORD].str.upper()
    twiss_df.index = twiss_df.index.str.upper()
    return twiss_df


def apply_correction(*lines: Line, correction: DataFrame) -> None:
    """Apply the given correction to the given lines.
    NOTE: The values given in the correction DataFrame are
    the KNL values, not the knob values.
    They need therefore to be divided by the length of the magnet/element.

    Args:
        corrections (DataFrame): Correction as calculated by 
        :func:`irnl_rdt_correction.main.irnl_rdt_correction`. 
    """        
    for _, (element, circuit, value) in correction[[NAME, CIRCUIT, VALUE]].iterrows():
        for line in lines:
            line.vars[circuit] = value / line.element_dict[element].length
