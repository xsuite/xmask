"""
Compact, readable Python reimplementation of corr_tripD1_v6.f.

Inputs (same as the Fortran code):
  temp/optics0_inser.mad   -- slice-by-slice optics: NAME BETX BETY X Y
  temp/tripD1D2.errors     -- MAD-X error table with K0L/K0SL/... columns

Output:
  temp/MCX_setting.mad     -- corrector settings (KQSX3, KCSSX3, ...)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path
from typing import Dict, List, Tuple


# ----- Configuration ---------------------------------------------------------
SIDES = ["L1", "R1", "L2", "R2", "L5", "R5", "L8", "R8"]  # ic = 1..8
MAG_TYPES = ["Q1", "Q2a", "Q2b", "Q3", "D1", "D2"]       # imag = 1..6

SCALE_MQX = 1.0
SCALE_D1 = 1.0
SCALE_D2 = 1.0

MAX_SLICES = 100


# ----- Data containers ------------------------------------------------------
@dataclass
class OpticsSlice:
    betx: float
    bety: float
    x: float
    y: float

    def res(self, i: int, j: int) -> float:
        """Return res(i,j) = betx^(i/2) * bety^(j/2)."""
        return (sqrt(self.betx) ** i) * (sqrt(self.bety) ** j)


@dataclass
class Accumulators:
    a2c: List[float] = field(default_factory=lambda: [0.0] * 8)
    a3c: List[List[float]] = field(default_factory=lambda: [[0.0] * 8 for _ in range(4)])
    a4c: List[List[float]] = field(default_factory=lambda: [[0.0] * 8 for _ in range(2)])
    a5c: List[List[float]] = field(default_factory=lambda: [[0.0] * 8 for _ in range(6)])
    a6c: List[List[float]] = field(default_factory=lambda: [[0.0] * 8 for _ in range(3)])
    b3c: List[List[float]] = field(default_factory=lambda: [[0.0] * 8 for _ in range(6)])
    b4c: List[List[float]] = field(default_factory=lambda: [[0.0] * 8 for _ in range(3)])
    b5c: List[List[float]] = field(default_factory=lambda: [[0.0] * 8 for _ in range(6)])
    b6c: List[List[float]] = field(default_factory=lambda: [[0.0] * 8 for _ in range(4)])

    # Slice-resolved optics weights for magnets (aux) and accumulated errors (b)
    a2aux: Dict[Tuple[int, int], List[float]] = field(default_factory=dict)  # (ic,imag) -> list over slices
    a3aux: Dict[Tuple[int, int], List[List[float]]] = field(default_factory=dict)  # 4 comps
    a4aux: Dict[Tuple[int, int], List[List[float]]] = field(default_factory=dict)  # 2 comps
    a5aux: Dict[Tuple[int, int], List[List[float]]] = field(default_factory=dict)  # 6 comps
    a6aux: Dict[Tuple[int, int], List[List[float]]] = field(default_factory=dict)  # 3 comps
    b3aux: Dict[Tuple[int, int], List[List[float]]] = field(default_factory=dict)  # 6 comps
    b4aux: Dict[Tuple[int, int], List[List[float]]] = field(default_factory=dict)  # 3 comps
    b5aux: Dict[Tuple[int, int], List[List[float]]] = field(default_factory=dict)  # 6 comps
    b6aux: Dict[Tuple[int, int], List[List[float]]] = field(default_factory=dict)  # 4 comps

    a2b: List[float] = field(default_factory=lambda: [0.0] * 8)
    a3b: List[List[float]] = field(default_factory=lambda: [[0.0] * 8 for _ in range(4)])
    a4b: List[List[float]] = field(default_factory=lambda: [[0.0] * 8 for _ in range(2)])
    a5b: List[List[float]] = field(default_factory=lambda: [[0.0] * 8 for _ in range(6)])
    a6b: List[List[float]] = field(default_factory=lambda: [[0.0] * 8 for _ in range(3)])
    b3b: List[List[float]] = field(default_factory=lambda: [[0.0] * 8 for _ in range(6)])
    b4b: List[List[float]] = field(default_factory=lambda: [[0.0] * 8 for _ in range(3)])
    b5b: List[List[float]] = field(default_factory=lambda: [[0.0] * 8 for _ in range(6)])
    b6b: List[List[float]] = field(default_factory=lambda: [[0.0] * 8 for _ in range(4)])


# ----- Helpers --------------------------------------------------------------
def side_index(name: str) -> int:
    for idx, tag in enumerate(SIDES):
        if tag in name:
            return idx
    raise ValueError(f"Cannot determine IR side for {name}")


def magnet_type(name: str) -> Tuple[int, float]:
    """Return (imag_index, scale)."""
    if ("MQXA.1" in name or "MQXC.1" in name or "MQXFA.A1" in name or "MQXFA.B1" in name):
        return 0, SCALE_MQX
    if ("MQXB.A" in name or "MQXD.A" in name or "MQXFB.A" in name
            or "MCBXFBH.A" in name or "MCBXFBV.A" in name):
        return 1, SCALE_MQX
    if ("MQXB.B" in name or "MQXD.B" in name or "MQXFB.B" in name
            or "MCBXFBH.B" in name or "MCBXFBV.B" in name):
        return 2, SCALE_MQX
    if ("MQXA.3" in name or "MQXC.3" in name or "MQXFA.A3" in name or "MQXFA.B3" in name
            or "MCBXFAH." in name or "MCBXFAV." in name):
        return 3, SCALE_MQX
    if "MBX" in name:
        return 4, SCALE_D1
    if ("MBRC." in name or "MBRD." in name):
        return 5, SCALE_D2
    raise ValueError(f"Cannot determine magnet type for {name}")


def ensure_lists(store: Dict[Tuple[int, int], List], key: Tuple[int, int], ncomp: int):
    if key not in store:
        store[key] = [[] for _ in range(ncomp)]
    return store[key]


# ----- Parsing optics -------------------------------------------------------
def read_optics(path: Path) -> Tuple[Accumulators, List[List[int]], int]:
    acc = Accumulators()
    slices_count = [[0] * 6 for _ in range(8)]  # per ic, imag
    bv = 0
    last_ic = None

    with path.open() as fh:
        for line in fh:
            if not line or line[0] in ("@", "*", "$", "#"):
                continue
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                name = parts[0]
                betx, bety, x, y = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                # Mimic Fortran's err=99 fallback on malformed lines.
                continue
            ic = side_index(name)
            if last_ic is None:
                last_ic = ic
            elif ic == last_ic + 1:
                bv += 1
            elif ic == last_ic - 1:
                bv -= 1
            last_ic = ic

            sl = OpticsSlice(betx, bety, x, y)

            # Dedicated correctors
            if "MQSX" in name or "MCQSX" in name:
                acc.a2c[ic] = sl.res(1, 1)
                continue
            if "MCSX" in name:
                acc.b3c[0][ic] = sl.res(3, 0)
                acc.b3c[1][ic] = sl.res(0, 3)
                acc.b3c[2][ic] = sl.res(1, 2)
                acc.b3c[3][ic] = sl.res(2, 1)
                acc.b3c[4][ic] = sl.res(1, 1) * y
                acc.b3c[5][ic] = 0.0
                continue
            if "MCTX" in name:
                acc.b6c[0][ic] = sl.res(6, 0)
                acc.b6c[1][ic] = sl.res(0, 6)
                acc.b6c[2][ic] = sl.res(4, 2)
                acc.b6c[3][ic] = sl.res(2, 4)
                continue
            if "MCOSX" in name:
                acc.a4c[0][ic] = sl.res(1, 3)
                acc.a4c[1][ic] = sl.res(3, 1)
                continue
            if "MCOX" in name:
                acc.b4c[0][ic] = sl.res(4, 0)
                acc.b4c[1][ic] = sl.res(0, 4)
                acc.b4c[2][ic] = sl.res(2, 2)
                continue
            if "MCSSX" in name:
                acc.a3c[0][ic] = sl.res(2, 1)
                acc.a3c[1][ic] = sl.res(1, 2)
                acc.a3c[2][ic] = sl.res(0, 3)
                acc.a3c[3][ic] = sl.res(3, 0)
                continue
            if "MCDSX" in name:
                acc.a5c[0][ic] = sl.res(4, 1)
                acc.a5c[1][ic] = sl.res(1, 4)
                acc.a5c[2][ic] = sl.res(2, 3)
                acc.a5c[3][ic] = sl.res(3, 2)
                acc.a5c[4][ic] = sl.res(0, 5)
                acc.a5c[5][ic] = sl.res(5, 0)
                continue
            if "MCDX" in name:
                acc.b5c[0][ic] = sl.res(5, 0)
                acc.b5c[1][ic] = sl.res(0, 5)
                acc.b5c[2][ic] = sl.res(3, 2)
                acc.b5c[3][ic] = sl.res(2, 3)
                acc.b5c[4][ic] = sl.res(1, 4)
                acc.b5c[5][ic] = sl.res(4, 1)
                continue
            if "MCTSX" in name:
                acc.a6c[0][ic] = sl.res(5, 1)
                acc.a6c[1][ic] = sl.res(1, 5)
                acc.a6c[2][ic] = sl.res(3, 3)
                continue

            # Main magnets
            imag, scale = magnet_type(name)
            slices_count[ic][imag] += 1
            if slices_count[ic][imag] > MAX_SLICES:
                raise RuntimeError(f"Too many slices in element {name}")
            res = sl

            a2_list = ensure_lists(acc.a2aux, (ic, imag), 1)
            a3_list = ensure_lists(acc.a3aux, (ic, imag), 4)
            a4_list = ensure_lists(acc.a4aux, (ic, imag), 2)
            a5_list = ensure_lists(acc.a5aux, (ic, imag), 6)
            a6_list = ensure_lists(acc.a6aux, (ic, imag), 3)
            b3_list = ensure_lists(acc.b3aux, (ic, imag), 6)
            b4_list = ensure_lists(acc.b4aux, (ic, imag), 3)
            b5_list = ensure_lists(acc.b5aux, (ic, imag), 6)
            b6_list = ensure_lists(acc.b6aux, (ic, imag), 4)

            a2_list[0].append(res.res(1, 1) * scale)
            a3_list[0].append(res.res(2, 1) * scale)
            a3_list[1].append(res.res(1, 2) * scale)
            a3_list[2].append(res.res(0, 3) * scale)
            a3_list[3].append(res.res(3, 0) * scale)
            a4_list[0].append(res.res(1, 3) * scale)
            a4_list[1].append(res.res(3, 1) * scale)
            a5_list[0].append(res.res(4, 1) * scale)
            a5_list[1].append(res.res(1, 4) * scale)
            a5_list[2].append(res.res(2, 3) * scale)
            a5_list[3].append(res.res(3, 2) * scale)
            a5_list[4].append(res.res(0, 5) * scale)
            a5_list[5].append(res.res(5, 0) * scale)
            a6_list[0].append(res.res(5, 1) * scale)
            a6_list[1].append(res.res(1, 5) * scale)
            a6_list[2].append(res.res(3, 3) * scale)
            b3_list[0].append(res.res(3, 0) * scale)
            b3_list[1].append(res.res(0, 3) * scale)
            b3_list[2].append(res.res(1, 2) * scale)
            b3_list[3].append(res.res(2, 1) * scale)
            b3_list[4].append(res.res(1, 1) * res.y * scale)
            b3_list[5].append(0.0)
            b4_list[0].append(res.res(4, 0) * scale)
            b4_list[1].append(res.res(0, 4) * scale)
            b4_list[2].append(res.res(2, 2) * scale)
            b5_list[0].append(res.res(5, 0) * scale)
            b5_list[1].append(res.res(0, 5) * scale)
            b5_list[2].append(res.res(3, 2) * scale)
            b5_list[3].append(res.res(2, 3) * scale)
            b5_list[4].append(res.res(1, 4) * scale)
            b5_list[5].append(res.res(4, 1) * scale)
            b6_list[0].append(res.res(6, 0) * scale)
            b6_list[1].append(res.res(0, 6) * scale)
            b6_list[2].append(res.res(4, 2) * scale)
            b6_list[3].append(res.res(2, 4) * scale)

    # Swap feed-down terms between left/right (b3 component 6) as in Fortran.
    for ic in range(0, 8, 2):
        acc.b3c[5][ic] = acc.b3c[4][ic + 1]
        acc.b3c[5][ic + 1] = acc.b3c[4][ic]
        for imag in range(6):
            left_key = (ic, imag)
            right_key = (ic + 1, imag)
            if left_key not in acc.b3aux or right_key not in acc.b3aux:
                continue
            left_y = acc.b3aux[left_key][4]
            right_y = acc.b3aux[right_key][4]
            if len(left_y) != len(right_y):
                raise RuntimeError(f"Feed-down slice mismatch ic={ic} imag={imag}")
            n = len(left_y)
            acc.b3aux[left_key][5] = [right_y[n - 1 - i] for i in range(n)]
            acc.b3aux[right_key][5] = [left_y[n - 1 - i] for i in range(n)]

    bv = 1 if bv > 0 else -1
    return acc, slices_count, bv


# ----- Parsing errors and accumulation --------------------------------------
def read_errors(path: Path, acc: Accumulators, slices_expected: List[List[int]]):
    slices_seen = [[0] * 6 for _ in range(8)]
    with path.open() as fh:
        for line in fh:
            if not line or line[0] in ("@", "*", "$", "#"):
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            name = parts[0].strip('"')
            values = [float(v) for v in parts[1:]]
            # akl(plane, order+1); plane 0=normal,1=skew
            def akl(plane: int, order: int) -> float:
                idx = 2 * order + plane  # plane: 0->normal,1->skew
                return values[idx] if idx < len(values) else 0.0

            ic = side_index(name)
            imag, _ = magnet_type(name)
            slices_seen[ic][imag] += 1
            s_idx = slices_seen[ic][imag] - 1

            # Sanity: slice counts must match optics
            if slices_seen[ic][imag] > slices_expected[ic][imag]:
                raise RuntimeError(f"Too many slices in errors for {name}")

            # Extract optics weights for that slice
            a2aux = acc.a2aux[(ic, imag)][0][s_idx]
            a3aux = [comp[s_idx] for comp in acc.a3aux[(ic, imag)]]
            a4aux = [comp[s_idx] for comp in acc.a4aux[(ic, imag)]]
            a5aux = [comp[s_idx] for comp in acc.a5aux[(ic, imag)]]
            a6aux = [comp[s_idx] for comp in acc.a6aux[(ic, imag)]]
            b3aux = [comp[s_idx] for comp in acc.b3aux[(ic, imag)]]
            b4aux = [comp[s_idx] for comp in acc.b4aux[(ic, imag)]]
            b5aux = [comp[s_idx] for comp in acc.b5aux[(ic, imag)]]
            b6aux = [comp[s_idx] for comp in acc.b6aux[(ic, imag)]]

            acc.a2b[ic] += a2aux * akl(1, 1)  # skew quad
            for i, aux in enumerate(a3aux):
                acc.a3b[i][ic] += aux * akl(1, 2)  # skew sext
            for i, aux in enumerate(a4aux):
                acc.a4b[i][ic] += aux * akl(1, 3)  # skew oct
            for i, aux in enumerate(a5aux):
                acc.a5b[i][ic] += aux * akl(1, 4)  # skew deca
            for i, aux in enumerate(a6aux):
                acc.a6b[i][ic] += aux * akl(1, 5)  # skew 14-pole
            for i, aux in enumerate(b3aux):
                acc.b3b[i][ic] += aux * akl(0, 2)  # normal sext
            for i, aux in enumerate(b4aux):
                acc.b4b[i][ic] += aux * akl(0, 3)  # normal oct
            for i, aux in enumerate(b5aux):
                acc.b5b[i][ic] += aux * akl(0, 4)  # normal deca
            for i, aux in enumerate(b6aux):
                acc.b6b[i][ic] += aux * akl(0, 5)  # normal 14-pole

    # Check matching slice counts
    for ic in range(8):
        for imag in range(6):
            if slices_seen[ic][imag] and slices_seen[ic][imag] != slices_expected[ic][imag]:
                raise RuntimeError(f"Optics/errors mismatch: ic={ic} imag={imag}")


# ----- Pairwise solver ------------------------------------------------------
def solve_pairwise(cL1, cR1, cL2, cR2, b1, b2):
    det = cL1 * cR2 - cR1 * cL2
    if det == 0.0:
        return 0.0, 0.0
    # Solve cL*kL + cR*kR + b = 0  -> cL*kL + cR*kR = -b.
    kL = (cR2 * b1 - cR1 * b2) / det
    kR = (-cL2 * b1 + cL1 * b2) / det
    return kL, kR


def solve_pairwise_diff(c1L, c1R, c2L, c2R, b1L, b1R, b2L, b2R):
    """
    For difference cancellations (antisymmetric): uses Fortran pattern
    det = -c1L*c2R + c1R*c2L
    aux1 = -b1L + b1R
    aux2 = -b2L + b2R
    kL = (-c2R*aux1 + c1R*aux2)/det
    kR = (-c2L*aux1 + c1L*aux2)/det
    """
    det = -c1L * c2R + c1R * c2L
    if det == 0.0:
        return 0.0, 0.0
    aux1 = -b1L + b1R
    aux2 = -b2L + b2R
    kL = (-c2R * aux1 + c1R * aux2) / det
    kR = (-c2L * aux1 + c1L * aux2) / det
    return kL, kR


# ----- Correction synthesis -------------------------------------------------
def build_settings(acc: Accumulators, bv: int) -> Dict[str, float]:
    out: Dict[str, float] = {}

    # A2 (local)
    for ic in range(8):
        out[f"KQSX3.{SIDES[ic]}"] = -acc.a2b[ic] / acc.a2c[ic]

    # A3 difference (0,3)/(3,0)
    for ic in range(0, 8, 2):
        kL, kR = solve_pairwise_diff(
            acc.a3c[2][ic], acc.a3c[2][ic + 1],
            acc.a3c[3][ic], acc.a3c[3][ic + 1],
            acc.a3b[2][ic], acc.a3b[2][ic + 1],
            acc.a3b[3][ic], acc.a3b[3][ic + 1],
        )
        out[f"KCSSX3.{SIDES[ic]}"] = bv * kL
        out[f"KCSSX3.{SIDES[ic+1]}"] = bv * kR

    # A4 sum (1,3)/(3,1)
    for ic in range(0, 8, 2):
        kL, kR = solve_pairwise(
            acc.a4c[0][ic], acc.a4c[0][ic + 1],
            acc.a4c[1][ic], acc.a4c[1][ic + 1],
            -(acc.a4b[0][ic] + acc.a4b[0][ic + 1]),
            -(acc.a4b[1][ic] + acc.a4b[1][ic + 1]),
        )
        out[f"KCOSX3.{SIDES[ic]}"] = kL
        out[f"KCOSX3.{SIDES[ic+1]}"] = kR

    # A5 difference (0,5)/(5,0) IR1/5 only
    for ic in (0, 4):  # L1/L5
        kL, kR = solve_pairwise_diff(
            acc.a5c[4][ic], acc.a5c[4][ic + 1],
            acc.a5c[5][ic], acc.a5c[5][ic + 1],
            acc.a5b[4][ic], acc.a5b[4][ic + 1],
            acc.a5b[5][ic], acc.a5b[5][ic + 1],
        )
        out[f"KCDSX3.{SIDES[ic]}"] = bv * kL
        out[f"KCDSX3.{SIDES[ic+1]}"] = bv * kR

    # A6 sum (5,1)/(1,5) IR1/5 only
    for ic in (0, 4):
        kL, kR = solve_pairwise(
            acc.a6c[0][ic], acc.a6c[0][ic + 1],
            acc.a6c[1][ic], acc.a6c[1][ic + 1],
            -(acc.a6b[0][ic] + acc.a6b[0][ic + 1]),
            -(acc.a6b[1][ic] + acc.a6b[1][ic + 1]),
        )
        out[f"KCTSX3.{SIDES[ic]}"] = kL
        out[f"KCTSX3.{SIDES[ic+1]}"] = kR

    # B3 difference (1,2)/(2,1)
    for ic in range(0, 8, 2):
        kL, kR = solve_pairwise_diff(
            acc.b3c[2][ic], acc.b3c[2][ic + 1],
            acc.b3c[3][ic], acc.b3c[3][ic + 1],
            acc.b3b[2][ic], acc.b3b[2][ic + 1],
            acc.b3b[3][ic], acc.b3b[3][ic + 1],
        )
        out[f"KCSX3.{SIDES[ic]}"] = kL
        out[f"KCSX3.{SIDES[ic+1]}"] = kR

    # B4 sum (0,4)/(4,0)
    for ic in range(0, 8, 2):
        kL, kR = solve_pairwise(
            acc.b4c[0][ic], acc.b4c[0][ic + 1],
            acc.b4c[1][ic], acc.b4c[1][ic + 1],
            -(acc.b4b[0][ic] + acc.b4b[0][ic + 1]),
            -(acc.b4b[1][ic] + acc.b4b[1][ic + 1]),
        )
        out[f"KCOX3.{SIDES[ic]}"] = bv * kL
        out[f"KCOX3.{SIDES[ic+1]}"] = bv * kR

    # B5 difference (5,0)/(0,5) IR1/5 only
    for ic in (0, 4):
        kL, kR = solve_pairwise_diff(
            acc.b5c[0][ic], acc.b5c[0][ic + 1],
            acc.b5c[1][ic], acc.b5c[1][ic + 1],
            acc.b5b[0][ic], acc.b5b[0][ic + 1],
            acc.b5b[1][ic], acc.b5b[1][ic + 1],
        )
        out[f"KCDX3.{SIDES[ic]}"] = kL
        out[f"KCDX3.{SIDES[ic+1]}"] = kR

    # B6 sum (6,0)/(0,6)
    for ic in range(0, 8, 2):
        kL, kR = solve_pairwise(
            acc.b6c[0][ic], acc.b6c[0][ic + 1],
            acc.b6c[1][ic], acc.b6c[1][ic + 1],
            -(acc.b6b[0][ic] + acc.b6b[0][ic + 1]),
            -(acc.b6b[1][ic] + acc.b6b[1][ic + 1]),
        )
        out[f"KCTX3.{SIDES[ic]}"] = bv * kL
        out[f"KCTX3.{SIDES[ic+1]}"] = bv * kR

    return out


# ----- Writer ---------------------------------------------------------------
def write_madx(settings: Dict[str, float], out_path: Path):
    def fmt(val: float) -> str:
        return f"{val: .6E}"

    def length_name(family: str, side: str) -> str:
        ir = side[:2]  # L1, R1, ...
        if family == "KQSX3":
            return "l.MQSXF" if ir in ("L1", "R1", "L5", "R5") else "l.MQSX"
        if family == "KCSSX3":
            return "l.MCSSXF" if ir in ("L1", "R1", "L5", "R5") else "l.MCSSX"
        if family == "KCOSX3":
            return "l.MCOSXF" if ir in ("L1", "R1", "L5", "R5") else "l.MCOSX"
        if family == "KCDSX3":
            return "l.MCDSXF"
        if family == "KCTSX3":
            return "l.MCTSXF"
        if family == "KCSX3":
            return "l.MCSXF" if ir in ("L1", "R1", "L5", "R5") else "l.MCSX"
        if family == "KCOX3":
            return "l.MCOXF" if ir in ("L1", "R1", "L5", "R5") else "l.MCOX"
        if family == "KCDX3":
            return "l.MCDXF"
        if family == "KCTX3":
            return "l.MCTXF" if ir in ("L1", "R1", "L5", "R5") else "l.MCTX"
        return f"l.{family}"

    order = [
        ("!! MQSX (a2) corrector", "KQSX3", SIDES),
        ("!! MCSSX (a3) corrector", "KCSSX3", SIDES),
        ("!! MCOSX (a4) corrector", "KCOSX3", SIDES),
        ("!! MCDSX (a5) corrector", "KCDSX3", ["L1", "R1", "L5", "R5"]),
        ("!! MCTSX (a6) corrector", "KCTSX3", ["L1", "R1", "L5", "R5"]),
        ("!! MCSX (b3) corrector", "KCSX3", SIDES),
        ("!! MCOX (b4) corrector", "KCOX3", SIDES),
        ("!! MCSX (b5) corrector", "KCDX3", ["L1", "R1", "L5", "R5"]),
        ("!! MCTX (b6) corrector", "KCTX3", SIDES),
    ]

    lines: List[str] = []
    for comment, family, sides in order:
        lines.append(f"{comment}\n")
        for side in sides:
            key = f"{family}.{side}"
            val = settings.get(key)
            if val is None:
                continue
            lines.append(f"{key} := {fmt(val)} / {length_name(family, side)};\n")
        lines.append("\n")
    lines.append("return;\n")
    out_path.write_text("".join(lines))


# ----- Main entry -----------------------------------------------------------
def main(
    optics_path: Path = Path("temp/optics0_inser.mad"),
    errors_path: Path = Path("temp/tripD1D2.errors"),
    output_path: Path = Path("temp/MCX_setting.mad"),
):
    acc, slices_expected, bv = read_optics(optics_path)
    read_errors(errors_path, acc, slices_expected)
    settings = build_settings(acc, bv)
    write_madx(settings, output_path)
    print(f"Wrote {output_path} with {len(settings)} assignments (bv={bv}).")


if __name__ == "__main__":
    main()
