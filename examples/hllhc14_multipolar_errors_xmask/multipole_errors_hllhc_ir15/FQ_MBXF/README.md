# Field Quality Data — D1 Magnets (MBXF)

This directory contains field quality multipole data for D1 magnets (MBXF series) under cold nominal measurement conditions (nominal current 12110 A).

## File Naming Convention

```
FQ_MBXF<ID>_cold_nominal.json
```

Where:
- `<ID>` is the magnet identifier (1, 2, 3, 5).
- MBXF1 is measured; MBXF2, MBXF3, and MBXF5 are extrapolated.

## File Format

Each JSON file contains the multipole field quality data for a single MBXF magnet.

```json
{
  "magnet": "MBXF1",
  "measurement": "cold_nominal",
  "reference_radius_mm": 50.0,
  "multipoles": [
    { "n": 2,  "bn": -0.26, "an": -1.09 },
    { "n": 3,  "bn":  2.33, "an":  1.8  },
    { "n": 4,  "bn":  0.1,  "an": -0.06 },
    { "n": 5,  "bn":  1.57, "an": -0.16 },
    { "n": 6,  "bn":  0.12, "an":  0.16 },
    { "n": 7,  "bn": -0.58, "an":  0.25 },
    { "n": 8,  "bn":  0.01, "an": -0.09 },
    { "n": 9,  "bn":  0.12, "an":  0.02 },
    { "n": 10, "bn": -0.01, "an": -0.01 },
    { "n": 11, "bn": -0.02, "an": -0.02 }
  ]
}
```

## Field Definitions

| Key                   | Meaning                                                   |
| --------------------- | --------------------------------------------------------- |
| `magnet`              | Magnet identifier (MBXF series)                           |
| `measurement`         | Measurement condition (`cold_nominal`)                    |
| `reference_radius_mm` | Reference radius used for multipole normalization (50 mm) |
| `multipoles`          | List of multipole coefficients for this magnet            |

All multipole values (bn, an) are given in units of 10⁻⁴ at the reference radius of 50 mm.
