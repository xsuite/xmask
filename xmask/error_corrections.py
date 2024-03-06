"""
Perform general corrections of optics errors.
"""
import json
from pathlib import Path
from typing import Any, Dict, Sequence, Union

import xtrack as xt

from xmask.lhc import correct_ir_rdts

ConfigType = Union[Dict[str, Any], str, Path]
Strings = Union[str, Sequence[str]]


def correct_errors(collider: xt.Multiline, 
    enable_ir_rdt_correction: bool = False, 
    ir_rdt_corr_config: ConfigType = None,
    ):
    """Correct the optics errors in the machine.
    Thus far, only the RDT correction in the IRs is implemented.

    Hint: Run orbit, tune and chroma corrections of :func:`xmask.tuning.machine_tuning` 
    before and after the corrections are applied. Run coupling correction only after, 
    as it will probably fail if run before the error-corrections are applied .

    Args:
        collider (xt.Multiline): XTrack MultiLine object to run the correction on.
        enable_ir_rdt_correction (bool, optional): Perform RDT correction in the IRs. Defaults to False.
        ir_rdt_corr_config (ConfigType, optional): Parameters for the RDT correction. Defaults to None.
    """
    print(f"Correcting Errors.")
    if enable_ir_rdt_correction:
        ir_rdt_correction(collider, config=ir_rdt_corr_config)


def ir_rdt_correction(collider: xt.Multiline, config: ConfigType):
    """ Loads the settings and performs the IR RDT correction.

    Args:
        collider (xt.Multiline): XTrack MultiLine object to run the correction on.
        ir_rdt_corr_config (ConfigType, optional): Parameters for the RDT correction. Defaults to None.
    """
    assert config is not None, "ir_rdt_corr_config is required"

    if isinstance(config, (str, Path)):
        with open(config, 'r') as fid:
            config = json.load(fid)
    else:
        config = config.copy()
    
    config.pop("enable", None)          # just to be safe

    assert "target_lines" in config
    target_lines = config.pop("target_lines")  # needs to be removed from config
    
    if isinstance(target_lines, str):  # I'll allow it
        target_lines = [target_lines]

    try:
        config["output"] = config["output"].format("".join(target_lines))
    except (KeyError, AttributeError):
        pass
    
    lines = [collider[line_name] for line_name in target_lines]
    beams = [1 if ln == 'lhcb1' else 4 for ln in target_lines]  # line lhcb2 is MAD-X lhcb4

    print(f"Correcting nonlinearities in the IRs via RDTs for {str(target_lines)}.")
    irnl_correction = correct_ir_rdts.calculate_correction(
        *lines, beams=beams, **config,
    )

    # If the lines are connected, it should be enough to apply the correction
    # to a single line, as the corrector magnets are common magnets.
    # But it is safer to just apply it to all lines.
    correct_ir_rdts.apply_correction(*lines, correction=irnl_correction)
    print()
