"""
Perform general corrections of optics errors.
"""
import json
from pathlib import Path
from typing import Any, Dict, Union

import xtrack as xt

from xmask.lhc import correct_ir_rdts

ConfigType = Union[Dict[str, Any], str, Path]


def correct_errors(line: xt.Line, 
    enable_ir_rdt_correction: bool = False, 
    ir_rdt_corr_config: ConfigType = None,
    ):
    """Correct the optics errors in the machine.
    Thus far, only the RDT correction in the IRs is implemented.

    Hint: Run orbit, tune and chroma corrections of :func:`xmask.tuning.machine_tuning` 
    before and after the corrections are applied. 
    Coupling correction on the other hand will probably fail.

    Args:
        line (Line): XTrack line object to run the correction on.
        enable_ir_rdt_correction (bool, optional): Perform RDT correction in the IRs. Defaults to False.
        ir_rdt_corr_config (ConfigType, optional): Parameters for the RDT correction. Defaults to None.
    """
    if enable_ir_rdt_correction:
        ir_rdt_correction(line, ir_rdt_corr_config)


def ir_rdt_correction(line: xt.Line, config: ConfigType):
    """ Loads the settings and performs the IR RDT correction.

    Args:
        line (Line): XTrack line object to run the correction on.
        ir_rdt_corr_config (ConfigType, optional): Parameters for the RDT correction. Defaults to None.
    """
    print('Correcting nonlinear errors in the IRs')
    assert config is not None, "ir_rdt_corr_config is required"

    if isinstance(config, (str, Path)):
        with open(config, 'r') as fid:
            config = json.load(fid)
    else:
        config = config.copy()
    
    line_name = config.pop('line_name')
    try:
        config["output"] = config["output"].format(line_name)
    except KeyError:
        pass

    irnl_correction = correct_ir_rdts.calculate_correction(
        line, 
        beams=[1 if line_name == 'lhcb1' else 4],  # line lhcb2 is MAD-X lhcb4
        **config,
    )
    correct_ir_rdts.apply_correction(line, correction=irnl_correction)
    print()
