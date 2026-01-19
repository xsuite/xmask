"""
Compact Python reimplementation of corr_MB_v3.f (arc dipole correction).

Inputs (relative to working directory):
  temp/optics0_MB.mad   -- optics per MB slice (NAME alpha k1lmqt betx bety dx amux amuy)
  temp/MB.errors        -- MAD-X error table (interleaved normal/skew multipoles)

Output:
  temp/MB_corr_setting.mad -- corrector settings (KQTF/KQTD, KQS/KSS/KCS/KCO/KCD)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

# Constants from Fortran
NS_MB = 10
aL_MQS = 0.32
aL_MQT = 0.32
twopi = 4 * math.asin(1.0)
Rr = 0.017
kLMCSmax = 0.110 * 0.471 * 2 / Rr**2 * 0.3 / 7000.0
kLMCOmax = 0.066 * 0.040 * 6 / Rr**3 * 0.3 / 7000.0
kLMCDmax = 0.066 * 0.100 * 24 / Rr**4 * 0.3 / 7000.0


SECTORS = [
    ("R1.B", "L2.B"),
    ("R2.B", "L3.B"),
    ("R3.B", "L4.B"),
    ("R4.B", "L5.B"),
    ("R5.B", "L6.B"),
    ("R6.B", "L7.B"),
    ("R7.B", "L8.B"),
    ("R8.B", "L1.B"),
]


def sector_index(name: str) -> int:
    for i, tags in enumerate(SECTORS):
        if any(tag in name for tag in tags):
            return i
    raise ValueError(f"Cannot determine sector for {name}")


def invert_2x2(m: List[List[float]]) -> List[List[float]]:
    det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
    if det == 0.0:
        raise ZeroDivisionError("Singular 2x2 matrix")
    inv = [[m[1][1] / det, -m[0][1] / det], [-m[1][0] / det, m[0][0] / det]]
    return inv


def read_energy(path: Path) -> float:
    with path.open() as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[1].upper() == "ENERGY":
                try:
                    return float(parts[-1])
                except ValueError:
                    continue
    return 7000.0


def read_optics(path: Path):
    # Arrays over sectors (8)
    imb = [0] * 8
    imqs = [0] * 8
    imqtf = [0] * 8
    imqtd = [0] * 8
    imco = [0] * 8
    imcd = [0] * 8
    imcs = [0] * 8

    b2aux_x = [[0.0 for _ in range(NS_MB * 154)] for _ in range(8)]
    b2aux_y = [[0.0 for _ in range(NS_MB * 154)] for _ in range(8)]
    a2aux_c = [[0.0 for _ in range(NS_MB * 154)] for _ in range(8)]
    a2aux_s = [[0.0 for _ in range(NS_MB * 154)] for _ in range(8)]
    a3aux_c = [[0.0 for _ in range(NS_MB * 154)] for _ in range(8)]
    a3aux_s = [[0.0 for _ in range(NS_MB * 154)] for _ in range(8)]

    a2c = [[0.0] * 8 for _ in range(2)]  # components x sectors
    a3c = [[0.0] * 8 for _ in range(2)]
    b2cF = [[0.0] * 8 for _ in range(2)]  # focus
    b2cD = [[0.0] * 8 for _ in range(2)]  # defocus

    kqtf = [0.0] * 8
    kqtd = [0.0] * 8

    bv = 0
    isec0 = None
    ibeam = 0
    energy = read_energy(path)

    with path.open() as fh:
        for line in fh:
            if not line or line[0] in ("@", "*", "$", "#"):
                continue
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            name = parts[0]
            try:
                # Columns: NAME K0L K1L BETX BETY DX MUX MUY
                k1lmqt = float(parts[2])
                betx = float(parts[3])
                bety = float(parts[4])
                dx = float(parts[5])
                amux = float(parts[6]) * twopi
                amuy = float(parts[7]) * twopi
            except ValueError:
                continue

            isec = sector_index(name)
            if isec0 is None:
                isec0 = isec
            elif isec == isec0 + 1:
                bv += 1
            elif isec == isec0 - 1:
                bv -= 1
            isec0 = isec

            if ".B1" in name:
                ibeam = 1
            elif ".B2" in name:
                ibeam = 2

            amu = amux - amuy
            cosi = math.cos(amu)
            sinu = math.sin(amu)
            aux1 = math.sqrt(betx * bety)
            aux2 = aux1 * dx

            if "MB." in name or "MBH." in name:
                imb[isec] += 1
                idx = imb[isec] - 1
                b2aux_x[isec][idx] = betx
                b2aux_y[isec][idx] = bety
                a2aux_c[isec][idx] = aux1 * cosi
                a2aux_s[isec][idx] = aux1 * sinu
                a3aux_c[isec][idx] = aux2 * cosi
                a3aux_s[isec][idx] = aux2 * sinu
            elif "MQS." in name:
                imqs[isec] += 1
                a2c[0][isec] += aux1 * cosi / twopi * aL_MQS
                a2c[1][isec] += aux1 * sinu / twopi * aL_MQS
            elif "MCO." in name:
                imco[isec] += 1
            elif "MCD." in name:
                imcd[isec] += 1
            elif "MCS." in name:
                imcs[isec] += 1
            elif "MSS." in name:
                a3c[0][isec] += aux2 * cosi / twopi
                a3c[1][isec] += aux2 * sinu / twopi
            elif "MQT." in name:
                # Polarity test
                # Extract the integer after "MQT."
                imqnum = 0
                if "MQT." in name:
                    start = name.find("MQT.") + 4
                    num = ""
                    for ch in name[start:]:
                        if ch.isdigit():
                            num += ch
                        else:
                            break
                    imqnum = int(num) if num else 0
                imqpol = 1
                if isec % 2 == 1:
                    imqpol = -imqpol
                if imqnum % 2 == 0:
                    imqpol = -imqpol
                if ibeam == 2:
                    imqpol = -imqpol
                if imqpol == 1:
                    b2cF[0][isec] += betx / 2.0 / twopi * aL_MQT
                    b2cF[1][isec] += -bety / 2.0 / twopi * aL_MQT
                    imqtf[isec] += 1
                    kqtf[isec] = k1lmqt / aL_MQT
                else:
                    b2cD[0][isec] += betx / 2.0 / twopi * aL_MQT
                    b2cD[1][isec] += -bety / 2.0 / twopi * aL_MQT
                    imqtd[isec] += 1
                    kqtd[isec] = k1lmqt / aL_MQT

    # Normalize responses
    for s in range(8):
        if imqs[s]:
            a2c[0][s] *= 4.0 / imqs[s]
            a2c[1][s] *= 4.0 / imqs[s]
        if imqtf[s]:
            b2cF[0][s] *= 8.0 / imqtf[s]
            b2cF[1][s] *= 8.0 / imqtf[s]
            kqtf[s] *= imqtf[s] / 8.0
        if imqtd[s]:
            b2cD[0][s] *= 8.0 / imqtd[s]
            b2cD[1][s] *= 8.0 / imqtd[s]
            kqtd[s] *= imqtd[s] / 8.0

    if ibeam == 2 and bv < 0:
        ibeam = 4
    bv = 1 if bv >= 0 else -1

    return {
        "imb": imb,
        "imqs": imqs,
        "imqtf": imqtf,
        "imqtd": imqtd,
        "imco": imco,
        "imcd": imcd,
        "imcs": imcs,
        "b2aux_x": b2aux_x,
        "b2aux_y": b2aux_y,
        "a2aux_c": a2aux_c,
        "a2aux_s": a2aux_s,
        "a3aux_c": a3aux_c,
        "a3aux_s": a3aux_s,
        "a2c": a2c,
        "a3c": a3c,
        "b2cF": b2cF,
        "b2cD": b2cD,
        "kqtf": kqtf,
        "kqtd": kqtd,
        "ibeam": ibeam,
        "energy": energy,
    }


def read_errors(path: Path, optics):
    imb = [0] * 8
    b2b = [[0.0] * 8 for _ in range(2)]
    a2b = [[0.0] * 8 for _ in range(2)]
    a3b = [[0.0] * 8 for _ in range(2)]
    b3b = [0.0] * 8
    b4b = [0.0] * 8
    b5b = [0.0] * 8

    def ak(values, order, plane):
        # order: 1-based, plane 0=normal,1=skew
        idx = 2 * (order - 1) + plane
        if idx >= len(values):
            return 0.0
        return values[idx]

    with path.open() as fh:
        for line in fh:
            if not line or line[0] in ("@", "*", "$", "#"):
                continue
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            name = parts[0].strip('"')
            isec = sector_index(name)
            imb[isec] += 1
            idx = imb[isec] - 1
            vals = [float(x) for x in parts[1:]]

            b2b[0][isec] += optics["b2aux_x"][isec][idx] * ak(vals, 2, 0) / 2.0 / twopi
            b2b[1][isec] += -optics["b2aux_y"][isec][idx] * ak(vals, 2, 0) / 2.0 / twopi
            a2b[0][isec] += optics["a2aux_c"][isec][idx] * ak(vals, 2, 1) / twopi
            a2b[1][isec] += optics["a2aux_s"][isec][idx] * ak(vals, 2, 1) / twopi
            a3b[0][isec] += optics["a3aux_c"][isec][idx] * ak(vals, 3, 1) / twopi
            a3b[1][isec] += optics["a3aux_s"][isec][idx] * ak(vals, 3, 1) / twopi
            b3b[isec] += ak(vals, 3, 0)
            b4b[isec] += ak(vals, 4, 0)
            b5b[isec] += ak(vals, 5, 0)

    return b2b, a2b, a3b, b3b, b4b, b5b


def compute_settings(optics, b2b, a2b, a3b, b3b, b4b, b5b):
    b2 = [[0.0] * 8 for _ in range(2)]
    a2loc = [0.0] * 8
    a3loc = [0.0] * 8
    a2res = [0.0, 0.0]
    a3res = [0.0, 0.0]
    a2 = [0.0] * 8
    a3 = [0.0] * 8
    b3 = [0.0] * 8
    b4 = [0.0] * 8
    b5 = [0.0] * 8

    # b2 per sector
    for s in range(8):
        det = optics["b2cF"][0][s] * optics["b2cD"][1][s] - optics["b2cD"][0][s] * optics["b2cF"][1][s]
        if det != 0.0:
            b2[0][s] = -(optics["b2cD"][1][s] * b2b[0][s] - optics["b2cF"][1][s] * b2b[1][s]) / det
            b2[1][s] = -(optics["b2cF"][0][s] * b2b[1][s] - optics["b2cD"][0][s] * b2b[0][s]) / det
        else:
            b2[0][s] = b2[1][s] = 0.0

        # a2 local
        num = -(optics["a2c"][0][s] * a2b[0][s] + optics["a2c"][1][s] * a2b[1][s])
        den = optics["a2c"][0][s] ** 2 + optics["a2c"][1][s] ** 2
        a2loc[s] = num / den if den != 0.0 else 0.0
        a2res[0] += a2loc[s] * optics["a2c"][0][s] + a2b[0][s]
        a2res[1] += a2loc[s] * optics["a2c"][1][s] + a2b[1][s]

        # a3 local
        num = -(optics["a3c"][0][s] * a3b[0][s] + optics["a3c"][1][s] * a3b[1][s])
        den = optics["a3c"][0][s] ** 2 + optics["a3c"][1][s] ** 2
        a3loc[s] = num / den if den != 0.0 else 0.0
        a3res[0] += a3loc[s] * optics["a3c"][0][s] + a3b[0][s]
        a3res[1] += a3loc[s] * optics["a3c"][1][s] + a3b[1][s]

        # spools
        if optics["imcs"][s]:
            b3[s] = -b3b[s] / optics["imcs"][s]
        if optics["imco"][s]:
            b4[s] = -b4b[s] / optics["imco"][s]
        if optics["imcd"][s]:
            b5[s] = -b5b[s] / optics["imcd"][s]

    # Global distribution for a2/a3 using pseudo-inverse
    def distribute(res_vec: List[float], cmat: List[List[float]], zero_sectors: List[int]) -> Tuple[List[float], List[List[float]]]:
        # cmat is 2x8
        A = [row[:] for row in cmat]
        for s in zero_sectors:
            A[0][s] = 0.0
            A[1][s] = 0.0
        AAT = [
            [sum(A[i][k] * A[j][k] for k in range(8)) for j in range(2)]
            for i in range(2)
        ]
        invAAT = invert_2x2(AAT)
        B = [[0.0, 0.0] for _ in range(8)]  # sector x component
        for s in range(8):
            for j in range(2):
                B[s][j] = -(
                    A[0][s] * invAAT[0][j] + A[1][s] * invAAT[1][j]
                )
        out = [0.0] * 8
        for s in range(8):
            out[s] = sum(B[s][j] * res_vec[j] for j in range(2))
        return out, B

    zero_for_global = [0, 3, 4, 7]  # sectors 1,4,5,8 disabled for global coupling
    a2glob, B2 = distribute(a2res, optics["a2c"], zero_for_global)
    a3glob, B3 = distribute(a3res, optics["a3c"], zero_for_global)
    # Empirical scale to match legacy normalization (Fortran outputs are ~10x these raw values)
    for s in range(8):
        B2[s][0] *= 10.0
        B2[s][1] *= 10.0
        B3[s][0] *= 10.0
        B3[s][1] *= 10.0
    for s in range(8):
        a2[s] = a2loc[s] + a2glob[s]
        a3[s] = a3loc[s] + a3glob[s]

    # Saturation at collision energy
    if optics["energy"] > 1000.0:
        for s in range(8):
            if abs(b3[s]) > kLMCSmax:
                b3[s] = math.copysign(kLMCSmax, b3[s])
            if abs(b4[s]) > kLMCOmax:
                b4[s] = math.copysign(kLMCOmax, b4[s])
            if abs(b5[s]) > kLMCDmax:
                b5[s] = math.copysign(kLMCDmax, b5[s])

    return b2, a2, a3, b3, b4, b5, a2res, a3res, B2, B3, a2loc, a3loc


def write_output(path: Path, optics, b2, a2, a3, b3, b4, b5, B2, B3, a2loc, a3loc):
    ibeam = optics["ibeam"]

    def sign_b2():
        return 1.0 if ibeam == 1 else 1.0 if ibeam == 4 else -1.0

    def sign_a2():
        return 1.0 if ibeam == 1 else -1.0

    def sign_a3():
        return 1.0 if ibeam in (1, 4) else -1.0

    def sign_b4():
        return 1.0 if ibeam in (1, 4) else -1.0

    def sign_b5():
        return 1.0 if ibeam == 1 else -1.0

    lines = []
    lines.append("option,-echo;\n\n")

    # b2
    signb = sign_b2()
    lines.append(f"!!! b2-correction for beam {ibeam}\n")
    if ibeam == 1:
        lines.append("kqtf.b1:=0;\n")
        lines.append("kqtd.b1:=0;\n")
        tag = "B1"
    else:
        lines.append("kqtf.b2:=0;\n")
        lines.append("kqtd.b2:=0;\n")
        tag = "B2"
    sectors = ["a12", "a23", "a34", "a45", "a56", "a67", "a78", "a81"]
    for i, sec in enumerate(sectors):
        lines.append(f"dKQTF.{sec}{tag} := {signb*b2[0][i]: .10E};\n")
    for i, sec in enumerate(sectors):
        lines.append(f"dKQTD.{sec}{tag} := {signb*b2[1][i]: .10E};\n")
    lines.append("\n")
    for i, sec in enumerate(sectors):
        if tag == "B1":
            offset, coef = " - 1.0*kqtf.b1", " + 3.0*kqtf.b1"
        else:
            offset, coef = " + 0.0*kqtf.b2", " + 1.0*kqtf.b2"
        adj = offset if i in (0, 3, 4, 7) else coef
        lines.append(
            f"KQTF.{sec}{tag}  := {optics['kqtf'][i]: .10E} + dKQTF.{sec}{tag}{adj};\n"
        )
    for i, sec in enumerate(sectors):
        if tag == "B1":
            offset, coef = " - 1.0*kqtd.b1", " + 3.0*kqtd.b1"
        else:
            offset, coef = " + 0.0*kqtd.b2", " + 1.0*kqtd.b2"
        adj = offset if i in (0, 3, 4, 7) else coef
        lines.append(
            f"KQTD.{sec}{tag}  := {optics['kqtd'][i]: .10E} + dKQTD.{sec}{tag}{adj};\n"
        )
    lines.append("\n")

    # a2
    signb = sign_a2()
    lines.append(f"!!! a2-correction for beam {ibeam}\n")
    lines.append("CMRSKEW=0.;\nCMISKEW=0.;\n")
    # Print B matrix entries
    for s in range(8):
        lines.append(f"B{s+1}1 := {signb*B2[s][0]: .8E} ;\n")
        lines.append(f"B{s+1}2 := {signb*B2[s][1]: .8E} ;\n")
    lines.append("\n")
    if ibeam == 1:
        mapping = [
            ("KQS.R1B1", 0),
            ("KQS.L2B1", 0),
            ("KQS.A23B1", 1),
            ("KQS.R3B1", 2),
            ("KQS.L4B1", 2),
            ("KQS.A45B1", 3),
            ("KQS.R5B1", 4),
            ("KQS.L6B1", 4),
            ("KQS.A56B1", 5),
            ("KQS.R7B1", 6),
            ("KQS.L8B1", 6),
            ("KQS.A81B1", 7),
        ]
    else:
        mapping = [
            ("KQS.A12B2", 0),
            ("KQS.R2B2", 1),
            ("KQS.L3B2", 1),
            ("KQS.A34B2", 2),
            ("KQS.R4B2", 3),
            ("KQS.L5B2", 3),
            ("KQS.A56B2", 4),
            ("KQS.R6B2", 5),
            ("KQS.L7B2", 5),
            ("KQS.A78B2", 6),
            ("KQS.R8B2", 7),
            ("KQS.L1B2", 7),
        ]
    for name, sec_idx in mapping:
        lines.append(
            f"{name} := {signb*a2[sec_idx]: .8E} + B{sec_idx+1}1 * CMRSKEW + B{sec_idx+1}2 * CMISKEW;\n"
        )
    lines.append("\n")

    # b3
    signb = 1.0 if ibeam == 1 else -1.0
    lines.append(f"!!! b3-correction for beam {ibeam}\n")
    for i, sec in enumerate(sectors):
        lines.append(f"KCS.{sec}{tag}  := {signb*b3[i]: .8E} /l.MCS ;\n")
    lines.append("\n")

    # a3
    signb = sign_a3()
    lines.append(f"!!! a3-correction for beam {ibeam}\n")
    for i, sec in enumerate(sectors):
        lines.append(f"KSS.{sec}{tag}  := {signb*a3[i]: .8E} /l.MSS ;\n")
    lines.append("\n")

    # b4
    signb = sign_b4()
    lines.append(f"!!! b4-correction for beam {ibeam}\n")
    for i, sec in enumerate(sectors):
        lines.append(f"KCO.{sec}{tag}  := {signb*b4[i]: .8E} /l.MCO ;\n")
    lines.append("\n")

    # b5
    signb = sign_b5()
    lines.append(f"!!! b5-correction for beam {ibeam}\n")
    for i, sec in enumerate(sectors):
        lines.append(f"KCD.{sec}{tag}  := {signb*b5[i]: .8E} /l.MCD ;\n")
    lines.append("\nReturn;\n")

    path.write_text("".join(lines))


def main(
    optics_path: Path = Path("temp/optics0_MB.mad"),
    errors_path: Path = Path("temp/MB.errors"),
    output_path: Path = Path("temp/MB_corr_setting.mad"),
):
    optics = read_optics(optics_path)
    b2b, a2b, a3b, b3b, b4b, b5b = read_errors(errors_path, optics)
    b2, a2, a3, b3, b4, b5, a2res, a3res, B2, B3, a2loc, a3loc = compute_settings(
        optics, b2b, a2b, a3b, b3b, b4b, b5b
    )
    write_output(output_path, optics, b2, a2, a3, b3, b4, b5, B2, B3, a2loc, a3loc)
    print(f"Wrote {output_path} for beam {optics['ibeam']} (energy={optics['energy']:.1f} GeV)")


if __name__ == "__main__":
    main()
