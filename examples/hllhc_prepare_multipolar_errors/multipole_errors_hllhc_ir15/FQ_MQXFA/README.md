# Field Quality Data — Q1/Q3 Magnets (MQXFA)

This directory contains field quality multipole data for Q1/Q3 magnets (MQXFA series) under cold nominal measurement conditions.

## File Naming Convention

```
MQXFA<ID>_CA<XX>.json
```

Where:
- `<ID>` is the magnet identifier (e.g., 03, 07b, 14b).
- `CA<XX>` is the cold mass assembly identifier, there are two magnets per CA.

## File Format

Each JSON file contains the multipole field quality data for a single MQXFA magnet.

```json
{
  "magnet": "MQXFA14b",
  "measurement": "cold_nominal",
  "reference_radius_mm": 50,
  "multipoles": [
    { "n": 3,  "bn": -0.12, "an": 1.19 },
    { "n": 4,  "bn": -0.59, "an": 0.89 },
    { "n": 5,  "bn":  3.24, "an": 1.53 },
    { "n": 6,  "bn":  2.03, "an": 3.05 },
    { "n": 7,  "bn": -0.30, "an": -0.19 },
    { "n": 8,  "bn":  0.12, "an": 0.65 },
    { "n": 9,  "bn":  0.06, "an": 0.04 },
    { "n": 10, "bn":  0.14, "an": -0.02 }
  ]
}
```

## Field Definitions

| Key                   | Meaning                                                   |
| --------------------- | --------------------------------------------------------- |
| `magnet`              | Magnet identifier (MQXFA series)                          |
| `measurement`         | Measurement condition (`cold_nominal`)                    |
| `reference_radius_mm` | Reference radius used for multipole normalization (50 mm) |
| `multipoles`          | List of multipole coefficients for this magnet            |

All multipole values (bn, an) are given in units of 10⁻⁴ at the reference radius of 50 mm.
