# Dataset: iPhone Battery Telemetry for ECC Model Validation

This dataset contains battery and system telemetry extracted from an iPhone
via Apple's `sysdiagnose` facility. It is used to calibrate and validate the
Enhanced Coulomb Counting (ECC) battery model described in our MCM 2026
Problem A paper.

**License:** CC BY 4.0 (see `LICENSE`).

**Device:** iPhone 16 Pro Max, iOS 18.3, 716 charge cycles, 86% state of
health (Q_ref = 3727 mAh, Q_design = 4329 mAh).

**Collection date:** 2026-02-01.

**Collection method:** `sysdiagnose` is a built-in iOS diagnostic facility
that records system telemetry in SQLite databases. No jailbreak or specialized
equipment is required. Any iPhone user can extract their own sysdiagnose
archive by following Apple's instructions.

---

## Directory Structure

```
data/
  sysdiagnose/
    CurrentPowerlog.PLSQL              53 MB   High-resolution powerlog (SQLite)
    powerlogs/
      log_*.EPSQL                     848 KB   Extended powerlog (SQLite)
  tables/                                      CSV exports of tables used by the model
    battery_state.csv                           Battery voltage, current, SOC, temperature
    display_events.csv                          Screen brightness and on/off state
    component_nodes.csv                         Hardware component identifiers
    component_energy.csv                        Per-component energy breakdown
    epsql_battery.csv                           5-min resolution battery data (38 days)
  parameters/                                  Model parameters derived from this data
    ocv_table.csv                               21-point OCV(z) lookup table
    ra_table.csv                                15-point R0(z) resistance table
    calibrated_params.json                      All calibrated model parameters
  README.md
  LICENSE
```

## Data Sources

### High-Resolution Powerlog (PLSQL)

**File:** `sysdiagnose/CurrentPowerlog.PLSQL`
**Format:** SQLite database (625 tables)
**Duration:** ~44 hours
**Resolution:** ~1 minute

Tables used by the analysis pipeline:

| Table | Rows | Description |
|-------|------|-------------|
| `PLBatteryAgent_EventBackward_Battery` | 6,905 | Battery state: SOC (%), terminal voltage (mV), instantaneous current (mA, negative = discharge), cell temperature, charging flag, cycle count, NominalChargeCapacity, OCV |
| `PLDisplayAgent_EventForward_Display` | 9,977 | Display brightness (normalized 0-1), millinits |
| `PLAccountingOperator_EventNone_Nodes` | 1,116 | Hardware component identifiers (CPU, GPU, Display, WiFi, Cellular, etc.) |
| `PLAccountingOperator_Aggregate_RootNodeEnergy` | 77,855 | Per-component and per-app energy consumption |

### Extended Powerlog (EPSQL)

**File:** `sysdiagnose/powerlogs/log_*.EPSQL`
**Format:** SQLite database
**Duration:** ~38 days
**Resolution:** ~5 minutes

| Table | Rows | Description |
|-------|------|-------------|
| `BatteryDataCollection_BDC_SBC_365_1` | 3,091 | Battery state: SOC, voltage, current, temperature, charging flag. Same fields as PLSQL but at lower resolution and longer history. |

**Note:** The EPSQL `InstantAmperage` field is a point-in-time snapshot rather
than an interval average, which introduces measurement noise for high-current
transients. The analysis pipeline clips the inferred CPU load to [0, 1] for
EPSQL data to compensate.

## Derived Parameters

### OCV Lookup Table (`parameters/ocv_table.csv`)

21-point piecewise-linear open-circuit voltage curve, extracted from near-rest
PLSQL rows (|InstantAmperage| < 15 mA, screen off, not charging). SOC bins
with sparse high-resolution data were supplemented by EPSQL BDC_SBC near-rest
measurements.

### Resistance Table (`parameters/ra_table.csv`)

15-point SOC-dependent internal resistance R0(z), read from the iOS
`RaTable` diagnostic field (716 cycles, 86% health). Values are in ohms.

### Calibrated Parameters (`parameters/calibrated_params.json`)

All model parameters used in the simulations, including:
- Battery cell parameters (capacity, resistance, thermal coefficients)
- Component power parameters (screen, CPU, WiFi, cellular, GPS, Bluetooth, audio, camera)
- System overhead values (screen-on: 230 mW, screen-off: 240 mW)
- Derived from the two-state overhead calibration described in the paper

## Reproducibility

To reproduce the analysis:

```bash
python analysis/sysdiagnose_tte.py --db data/sysdiagnose/CurrentPowerlog.PLSQL
```

The pipeline reads both PLSQL and EPSQL databases, calibrates the model,
runs forward and current-driven validation, Monte Carlo TTE, and sensitivity
analyses, producing 12 figures in `figures/`.

## Privacy Note

The sysdiagnose data was collected from the first author's personal device
with informed consent. The data contains no personally identifiable
information beyond aggregate app usage categories (e.g., "social media",
"video streaming") and hardware telemetry. No location data, message
content, or contact information is included.

## Citation

If you use this dataset, please cite our MCM 2026 paper and this repository:

```
@misc{miller2026battery,
  author = {Miller, Mark},
  title  = {{iPhone Battery Telemetry Dataset for ECC Model Validation}},
  year   = {2026},
  url    = {https://github.com/markm39/mcm-battery-2026}
}
```
