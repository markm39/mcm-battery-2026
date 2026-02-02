"""
Sysdiagnose-based TTE analysis pipeline.

Imports battery_model.py (paper-exact model) and parses a sysdiagnose
powerlog SQLite database to perform:

  (a) Component energy breakdown        -> Fig sysdiag_component_energy.png
  (b) App category power stacked bars   -> Fig sysdiag_app_categories.png
  (c) Forward sim vs observed SOC       -> Fig sysdiag_soc_comparison.png
  (d) Model vs observed voltage         -> Fig sysdiag_voltage_comparison.png
  (e) Current-driven validation
  (f) Monte Carlo TTE histogram         -> Fig sysdiag_mc_tte.png
  (g) Tornado sensitivity chart         -> Fig sysdiag_tornado.png

Usage:
    python analysis/sysdiagnose_tte.py [--db PATH] [--figures-dir PATH]

Default database path: data/raw/sysdiagnose/CurrentPowerlog.PLSQL
Default figures dir:   figures/
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root for battery_model import
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
import battery_model as bm


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DB = _PROJECT_ROOT / "data" / "raw" / "sysdiagnose" / "CurrentPowerlog.PLSQL"
DEFAULT_FIGURES = _PROJECT_ROOT / "figures"

# iOS hardware root-node name -> model component mapping
# These are the permanent hardware nodes in PLAccountingOperator_EventNone_Nodes
NODE_COMPONENT_MAP: Dict[str, str] = {
    "CPU":          "CPU",
    "GPU":          "GPU",
    "DRAM":         "CPU",
    "RestOfSOC":    "CPU",
    "VENC":         "CPU",
    "VDEC":         "CPU",
    "Display":      "Screen",
    "BB":           "Cellular",
    "BB-Voice":     "Cellular",
    "WiFi-Data":    "WiFi",
    "WiFi-Location": "WiFi",
    "WiFi-Pipeline": "WiFi",
    "GPS":          "GPS",
    "Nfc":          "Other",
    "AudioHeadset": "Audio",
    "AudioSpeaker": "Audio",
    "FrontCamera":  "Camera",
    "BackCamera":   "Camera",
    "ISP":          "Camera",
    "Torch":        "Camera",
    "Accessory":    "Other",
}

# App categories for grouping (bundle-id prefixes)
APP_CATEGORIES: Dict[str, List[str]] = {
    "Video":     ["com.apple.mobileslideshow", "com.google.ios.youtube",
                  "com.apple.tv", "com.netflix.Netflix"],
    "Social":    ["com.facebook.Facebook", "com.burbn.instagram",
                  "com.atebits.Tweetie2", "com.toyopagroup.picaboo",
                  "com.reddit.Reddit"],
    "Music":     ["com.apple.Music", "com.spotify.client",
                  "com.apple.podcasts"],
    "Nav":       ["com.apple.Maps", "com.google.Maps",
                  "com.waze.iphone"],
    "Messaging": ["com.apple.MobileSMS", "com.apple.mobilemail",
                  "org.whatsapp.WhatsApp", "com.facebook.Messenger",
                  "ph.telegra.Telegraph"],
    "Browser":   ["com.apple.mobilesafari", "com.google.chrome.ios"],
}

# Hardware-level node names (excluded from app breakdown)
_HW_NODES = set(NODE_COMPONENT_MAP.keys()) | {"__GLOBAL__"}


# ─────────────────────────────────────────────────────────────────────────────
# Database helpers
# ─────────────────────────────────────────────────────────────────────────────

def open_powerlog(db_path: Path) -> sqlite3.Connection:
    """Open the powerlog database and return a connection."""
    if not db_path.exists():
        raise FileNotFoundError(
            f"Powerlog database not found at {db_path}\n"
            f"To use this script:\n"
            f"  1. Run sysdiagnose on your iPhone\n"
            f"  2. Extract and symlink/copy the .PLSQL file to:\n"
            f"     {db_path}\n"
            f"  3. Re-run this script\n"
        )
    return sqlite3.connect(str(db_path))


def list_tables(conn: sqlite3.Connection) -> List[str]:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return [row[0] for row in cur.fetchall()]


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    tables = list_tables(conn)
    return name in tables


# ─────────────────────────────────────────────────────────────────────────────
# 2a. Query battery table
# ─────────────────────────────────────────────────────────────────────────────

def query_battery_table(conn: sqlite3.Connection) -> pd.DataFrame:
    """Query the main battery event table.

    Timestamps are Unix epoch floats. InstantAmperage is in mA
    (negative = discharge in iOS convention). Voltage is in mV.
    Temperature is in degrees C. Level is 0-100%.
    """
    table = "PLBatteryAgent_EventBackward_Battery"
    if not _has_table(conn, table):
        table = "PLBatteryAgent_EventBackward_BatteryUI"

    cols = ("timestamp, Level, InstantAmperage, Voltage, Temperature, "
            "IsCharging, ExternalConnected, CycleCount, "
            "NominalChargeCapacity, OCV")

    df = pd.read_sql_query(
        f"SELECT {cols} FROM [{table}] ORDER BY timestamp", conn)

    # Convert Unix epoch to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)  # drop tz for arithmetic
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Query display table
# ─────────────────────────────────────────────────────────────────────────────

def query_display_table(conn: sqlite3.Connection) -> Optional[pd.DataFrame]:
    """Query display events. Returns Brightness normalized to 0-1."""
    table = "PLDisplayAgent_EventForward_Display"
    if not _has_table(conn, table):
        return None

    df = pd.read_sql_query(
        f"SELECT timestamp, Brightness, mNits FROM [{table}] ORDER BY timestamp",
        conn)

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Normalize brightness to 0-1 using mNits (millinits)
    # Typical iPhone max: ~2000 nits outdoor, ~1000 nits indoor
    # mNits max ~2000000, but indoor max is ~1000000
    if "mNits" in df.columns:
        mnits = pd.to_numeric(df["mNits"], errors="coerce").fillna(0)
        max_mnits = mnits.quantile(0.99) if len(mnits) > 10 else 1e6
        if max_mnits < 1:
            max_mnits = 1e6
        df["brightness_01"] = np.clip(mnits / max_mnits, 0.0, 1.0)
    else:
        # Fallback: normalize raw Brightness by its max
        raw = pd.to_numeric(df["Brightness"], errors="coerce").fillna(0)
        mx = raw.quantile(0.99) if len(raw) > 10 else 1.0
        if mx < 0.001:
            mx = 1.0
        df["brightness_01"] = np.clip(raw / mx, 0.0, 1.0)

    # Infer screen on/off: brightness > 0 means on
    df["screen_on"] = df["brightness_01"] > 0.005

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Query node energy (component breakdown)
# ─────────────────────────────────────────────────────────────────────────────

def query_component_energy(conn: sqlite3.Connection) -> Optional[pd.DataFrame]:
    """Query hardware component energy by JOINing Nodes with RootNodeEnergy.

    Returns DataFrame with columns: NodeName, Component, Energy.
    Energy units are the raw powerlog units (typically millijoules or
    unitless energy counters -- we treat them as relative for comparison).
    """
    nodes_table = "PLAccountingOperator_EventNone_Nodes"
    energy_table = "PLAccountingOperator_Aggregate_RootNodeEnergy"

    if not _has_table(conn, nodes_table) or not _has_table(conn, energy_table):
        return None

    query = f"""
        SELECT n.Name AS NodeName, SUM(e.Energy) AS Energy
        FROM [{energy_table}] e
        JOIN [{nodes_table}] n ON e.RootNodeID = n.ID
        WHERE n.IsPermanent = 1
        GROUP BY n.Name
        ORDER BY Energy DESC
    """
    df = pd.read_sql_query(query, conn)
    if len(df) == 0:
        return None

    # Map node names to model components
    df["Component"] = df["NodeName"].map(
        lambda n: NODE_COMPONENT_MAP.get(n, "Other"))

    return df


def query_app_energy(conn: sqlite3.Connection) -> Optional[pd.DataFrame]:
    """Query per-app (leaf node) energy from RootNodeEnergy.

    Returns top apps by total energy with their names.
    """
    nodes_table = "PLAccountingOperator_EventNone_Nodes"
    energy_table = "PLAccountingOperator_Aggregate_RootNodeEnergy"

    if not _has_table(conn, nodes_table) or not _has_table(conn, energy_table):
        return None

    query = f"""
        SELECT n.Name AS AppName, SUM(e.Energy) AS Energy
        FROM [{energy_table}] e
        JOIN [{nodes_table}] n ON e.NodeID = n.ID
        WHERE n.IsPermanent = 0
        GROUP BY n.Name
        HAVING SUM(e.Energy) > 0
        ORDER BY Energy DESC
        LIMIT 50
    """
    df = pd.read_sql_query(query, conn)
    return df if len(df) > 0 else None


# ─────────────────────────────────────────────────────────────────────────────
# Query EPSQL (BatteryDataCollection) -- 38 days, 5-min resolution
# ─────────────────────────────────────────────────────────────────────────────

def find_epsql(db_dir: Path) -> Optional[Path]:
    """Find the EPSQL database in the powerlogs directory."""
    powerlogs = db_dir.parent / "powerlogs" if db_dir.name.endswith(".PLSQL") else db_dir
    if not powerlogs.is_dir():
        powerlogs = db_dir.parent
    for f in sorted(powerlogs.glob("*.EPSQL")):
        return f
    return None


def query_epsql_battery(epsql_path: Path) -> Optional[pd.DataFrame]:
    """Query the BDC SBC table from EPSQL (5-min interval data, ~38 days).

    Returns DataFrame with same column conventions as PLSQL battery table:
    Level (0-100), Voltage (mV), InstantAmperage (mA), Temperature (C),
    IsCharging (0/1), timestamp (datetime).
    """
    conn = sqlite3.connect(str(epsql_path))
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]

    sbc_table = None
    for t in tables:
        if "BDC_SBC" in t:
            sbc_table = t
            break
    if sbc_table is None:
        conn.close()
        return None

    df = pd.read_sql_query(
        f"SELECT timestamp, StateOfCharge, Voltage, InstantAmperage, "
        f"Temperature, IsCharging, AccumulatedSystemLoad "
        f"FROM [{sbc_table}] ORDER BY timestamp",
        conn)
    conn.close()

    if len(df) == 0:
        return None

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Rename to match PLSQL conventions
    df = df.rename(columns={"StateOfCharge": "Level"})

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2d. Extract discharge sessions
# ─────────────────────────────────────────────────────────────────────────────

def extract_discharge_sessions(
        batt_df: pd.DataFrame,
        min_soc_drop: float = 20.0
) -> List[pd.DataFrame]:
    """Split battery data into discharge sessions.

    A session is a continuous run of IsCharging=0 rows.
    Only sessions with >= min_soc_drop percentage points are kept.
    """
    if "IsCharging" not in batt_df.columns or "Level" not in batt_df.columns:
        return []

    df = batt_df.copy()
    # Mark transitions: group consecutive discharging rows
    df["_grp"] = (df["IsCharging"] != df["IsCharging"].shift(1)).cumsum()

    sessions = []
    for _, grp_df in df[df["IsCharging"] == 0].groupby("_grp"):
        if len(grp_df) < 5:
            continue
        soc_drop = grp_df["Level"].iloc[0] - grp_df["Level"].iloc[-1]
        if soc_drop >= min_soc_drop:
            sessions.append(grp_df.drop(columns=["_grp"]).reset_index(drop=True))

    return sessions


def find_main_session(sessions: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Find the deepest/longest discharge session."""
    if not sessions:
        return None
    return max(sessions, key=lambda s: (
        s["Level"].iloc[0] - s["Level"].iloc[-1]) * len(s))


# ─────────────────────────────────────────────────────────────────────────────
# 2b. Component energy breakdown (Figure 1)
# ─────────────────────────────────────────────────────────────────────────────

def compute_component_breakdown(
        node_df: Optional[pd.DataFrame]
) -> Optional[pd.DataFrame]:
    """Aggregate node energy into model component categories."""
    if node_df is None or "Component" not in node_df.columns:
        return None

    grouped = (node_df.groupby("Component", as_index=False)["Energy"]
               .sum()
               .sort_values("Energy", ascending=False))
    # Drop __GLOBAL__ and zero entries
    grouped = grouped[grouped["Energy"] > 0]
    grouped = grouped[grouped["Component"] != "Other"]
    return grouped if len(grouped) > 0 else None


def plot_component_energy(breakdown_df: pd.DataFrame, fig_path: Path) -> None:
    """Figure 1: Horizontal bar chart of component energy."""
    fig, ax = plt.subplots(figsize=(8, 5))

    components = breakdown_df["Component"].values
    energy = breakdown_df["Energy"].values
    total = energy.sum()
    pct = energy / total * 100

    colors = plt.cm.Set2(np.linspace(0, 1, len(components)))
    bars = ax.barh(components, pct, color=colors)
    ax.set_xlabel("Energy Share (%)")
    ax.set_title("Component Energy Breakdown (Sysdiagnose)")
    ax.invert_yaxis()

    for bar, p in zip(bars, pct):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{p:.1f}%", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {fig_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2c. App category stacked bars (Figure 2)
# ─────────────────────────────────────────────────────────────────────────────

def compute_app_categories(
        app_df: Optional[pd.DataFrame]
) -> Optional[pd.DataFrame]:
    """Group apps by category."""
    if app_df is None or "AppName" not in app_df.columns:
        return None

    records = []
    for _, row in app_df.iterrows():
        app_name = str(row["AppName"])
        energy = float(row.get("Energy", 0))

        category = "Other"
        for cat, bundle_ids in APP_CATEGORIES.items():
            if any(bid in app_name for bid in bundle_ids):
                category = cat
                break
        records.append({"Category": category, "AppName": app_name, "Energy": energy})

    df = pd.DataFrame(records)
    grouped = (df.groupby("Category", as_index=False)["Energy"]
               .sum()
               .sort_values("Energy", ascending=False))
    return grouped if len(grouped) > 0 else None


def plot_app_categories(cat_df: pd.DataFrame, fig_path: Path) -> None:
    """Figure 2: App category energy bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))

    categories = cat_df["Category"].values
    energy = cat_df["Energy"].values
    total = energy.sum()
    pct = energy / total * 100

    colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
    bars = ax.barh(categories, pct, color=colors)
    ax.set_xlabel("Energy Share (%)")
    ax.set_title("App Category Energy (Sysdiagnose)")
    ax.invert_yaxis()

    for bar, p in zip(bars, pct):
        if p > 1.0:
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{p:.1f}%", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {fig_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2e. Build usage schedule from sysdiagnose events
# ─────────────────────────────────────────────────────────────────────────────

def build_usage_schedule(
        session: pd.DataFrame,
        display_df: Optional[pd.DataFrame] = None,
        params: Optional[bm.BatteryParams] = None,
) -> bm.Callable[[float], Dict]:
    """Build a piecewise-constant usage schedule from sysdiagnose events.

    Power calibration: cpu_load is set so model total_power matches
    measured V*|I| at each interval. This keeps the model's equation
    pipeline intact while feeding it realistic power.

    Screen state:
      - If display_df provided: real brightness from PLDisplayAgent
      - Otherwise: inferred from |I| > 80mA heuristic
    """
    if params is None:
        params = bm.DEFAULT
    t0 = session["timestamp"].iloc[0]
    session_seconds = (session["timestamp"] - t0).dt.total_seconds().values

    # --- Screen state ---
    brightness_times: np.ndarray = np.array([])
    brightness_vals: np.ndarray = np.array([])
    screen_on_flags: np.ndarray = np.array([])

    if display_df is not None and "brightness_01" in display_df.columns:
        t_end = session["timestamp"].iloc[-1]
        mask = (display_df["timestamp"] >= t0) & (display_df["timestamp"] <= t_end)
        disp_session = display_df[mask]
        if len(disp_session) > 0:
            brightness_times = (
                disp_session["timestamp"] - t0).dt.total_seconds().values
            brightness_vals = disp_session["brightness_01"].values.astype(float)
            screen_on_flags = disp_session["screen_on"].values.astype(float)

    has_display = len(brightness_times) > 0

    # --- Power-calibrated CPU load ---
    # Compute measured battery power from V*|I| per interval
    voltage_v = pd.to_numeric(
        session["Voltage"], errors="coerce").fillna(3800).values / 1000.0
    current_ma = pd.to_numeric(
        session["InstantAmperage"], errors="coerce").fillna(0).values
    abs_current = np.abs(current_ma)
    measured_mw = voltage_v * abs_current

    # For each interval, solve for cpu_load that reproduces measured power
    n = len(session_seconds)
    cpu_loads = np.zeros(n)

    # Pre-compute screen state per battery interval for power calibration
    per_interval_screen_on = np.zeros(n, dtype=bool)
    per_interval_brightness = np.full(n, 0.5)

    for k in range(n):
        if has_display:
            didx = np.searchsorted(brightness_times,
                                   session_seconds[k], side="right") - 1
            didx = max(0, min(didx, len(brightness_vals) - 1))
            per_interval_brightness[k] = brightness_vals[didx]
            per_interval_screen_on[k] = (
                bool(screen_on_flags[didx]) if len(screen_on_flags) > 0
                else brightness_vals[didx] > 0.01)
        else:
            # Heuristic: >80mA suggests screen on
            per_interval_screen_on[k] = abs_current[k] > 80.0
            per_interval_brightness[k] = 0.5 if per_interval_screen_on[k] else 0.0

        son = per_interval_screen_on[k]
        br = per_interval_brightness[k]

        # Base power (everything except cpu_load contribution)
        p_base = 0.0
        if son:
            p_base += params.screen_base_mw + params.screen_brightness_coeff_mw * br
            p_base += params.overhead_screen_on_mw
        else:
            p_base += params.overhead_screen_off_mw
        p_base += params.wifi_idle_mw
        p_base += params.cellular_idle_mw
        p_base += params.bt_idle_mw

        if measured_mw[k] < params.cpu_idle_mw + 10:
            cpu_loads[k] = 0.0
        else:
            load = (measured_mw[k] - p_base) / params.cpu_load_coeff_mw
            cpu_loads[k] = float(max(load, 0.0))

    def schedule(t: float) -> Dict:
        batt_idx = np.searchsorted(session_seconds, t, side="right") - 1
        batt_idx = max(0, min(batt_idx, n - 1))

        return {
            "screen_on": bool(per_interval_screen_on[batt_idx]),
            "brightness": float(per_interval_brightness[batt_idx]),
            "cpu_load": float(cpu_loads[batt_idx]),
            "wifi_state": "idle",
            "wifi_data_rate": 0.0,
            "cellular_state": "idle",
            "cellular_signal": 0.8,
            "cellular_data_rate": 0.0,
            "gps_state": "off",
            "bluetooth_state": "idle",
            "audio_state": "off",
            "camera_state": "off",
        }

    return schedule


# ─────────────────────────────────────────────────────────────────────────────
# 2e-f. Forward simulation comparison (Figures 3-4)
# ─────────────────────────────────────────────────────────────────────────────

def forward_simulation(
        session: pd.DataFrame,
        display_df: Optional[pd.DataFrame],
        params: bm.BatteryParams,
        cycle_count: int = 716
) -> Dict:
    """Run forward simulation against a discharge session."""
    t0 = session["timestamp"].iloc[0]
    t_seconds = (session["timestamp"] - t0).dt.total_seconds().values
    t_max = float(t_seconds[-1])

    obs_soc = session["Level"].values.astype(float) / 100.0

    # Voltage: powerlog stores mV, convert to V
    obs_voltage = None
    if "Voltage" in session.columns:
        v_raw = pd.to_numeric(session["Voltage"], errors="coerce").values
        obs_voltage = v_raw / 1000.0  # mV -> V

    soc_init = float(obs_soc[0])

    T_init = 25.0
    if "Temperature" in session.columns:
        temps = pd.to_numeric(
            session["Temperature"], errors="coerce").dropna().values
        if len(temps) > 0:
            T_init = float(temps[0])
            # Sanity check: if temp > 100, it might be centi-degrees
            if T_init > 80:
                T_init = T_init / 100.0

    usage_fn = build_usage_schedule(session, display_df, params)

    result = bm.simulate(
        usage_fn, params,
        soc_init=soc_init,
        T_ambient=T_init,
        T_cell_init=T_init,
        cycle_count=cycle_count,
        t_max=t_max,
        dt=60.0,
    )

    return {
        "model": result,
        "obs_t": t_seconds,
        "obs_soc": obs_soc,
        "obs_voltage": obs_voltage,
    }


def plot_soc_comparison(fwd: Dict, fig_path: Path) -> None:
    """Figure 3: Model vs observed SOC over time."""
    fig, ax = plt.subplots(figsize=(10, 5))

    t_obs_h = fwd["obs_t"] / 3600.0
    t_mod_h = fwd["model"]["t"] / 3600.0

    ax.plot(t_obs_h, fwd["obs_soc"] * 100, "k-", linewidth=2,
            label="Observed (iOS)")
    ax.plot(t_mod_h, fwd["model"]["soc"] * 100, "r--", linewidth=1.5,
            label="Model")

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("SOC (%)")
    ax.set_title("Model vs Observed SOC (Sysdiagnose)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {fig_path}")


def plot_voltage_comparison(fwd: Dict, fig_path: Path) -> None:
    """Figure 4: Model vs observed voltage over time."""
    if fwd["obs_voltage"] is None:
        print("  Skipping voltage plot (no voltage data)")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    t_obs_h = fwd["obs_t"] / 3600.0
    t_mod_h = fwd["model"]["t"] / 3600.0

    ax.plot(t_obs_h, fwd["obs_voltage"], "k-", linewidth=2, label="Observed")
    ax.plot(t_mod_h, fwd["model"]["voltage"], "r--", linewidth=1.5,
            label="Model")

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title("Model vs Observed Terminal Voltage (Sysdiagnose)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {fig_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2g. Current-driven validation
# ─────────────────────────────────────────────────────────────────────────────

def current_driven_validation(
        session: pd.DataFrame,
        params: bm.BatteryParams,
        cycle_count: int = 716
) -> Optional[Dict]:
    """Run current-driven simulation using measured InstantAmperage.

    iOS convention: InstantAmperage is negative during discharge.
    Model convention: positive = discharge.
    """
    if "InstantAmperage" not in session.columns:
        return None

    t0 = session["timestamp"].iloc[0]
    t_seconds = (session["timestamp"] - t0).dt.total_seconds().values

    # iOS: negative = discharge; flip to positive for model
    raw_ma = pd.to_numeric(
        session["InstantAmperage"], errors="coerce").fillna(0).values
    current_a = -raw_ma / 1000.0  # mA -> A, flip sign

    obs_soc = session["Level"].values.astype(float) / 100.0
    soc_init = float(obs_soc[0])

    T_init = 25.0
    if "Temperature" in session.columns:
        temps = pd.to_numeric(
            session["Temperature"], errors="coerce").dropna().values
        if len(temps) > 0:
            T_init = float(temps[0])
            if T_init > 80:
                T_init = T_init / 100.0

    def current_fn(t: float) -> float:
        idx = np.searchsorted(t_seconds, t, side="right") - 1
        idx = max(0, min(idx, len(current_a) - 1))
        return float(current_a[idx])

    t_max = float(t_seconds[-1])

    result = bm.simulate_current_driven(
        current_fn, params,
        soc_init=soc_init,
        T_ambient=T_init,
        T_cell_init=T_init,
        cycle_count=cycle_count,
        t_max=t_max,
        dt=60.0,
    )

    return {
        "model": result,
        "obs_t": t_seconds,
        "obs_soc": obs_soc,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2h. Monte Carlo TTE (Figures 5-6)
# ─────────────────────────────────────────────────────────────────────────────

def run_monte_carlo(
        session: pd.DataFrame,
        display_df: Optional[pd.DataFrame],
        params: bm.BatteryParams,
        cycle_count: int = 716,
        n_samples: int = 500
) -> Dict:
    """Run Monte Carlo TTE with parameter perturbation."""
    usage_fn = build_usage_schedule(session, display_df, params)

    print(f"  Running Monte Carlo TTE (N={n_samples})...")
    mc = bm.monte_carlo_tte(
        usage_fn, params, N=n_samples,
        cycle_count=cycle_count, dt=60.0,
    )

    # Observed TTE from session duration
    t0 = session["timestamp"].iloc[0]
    t_end = session["timestamp"].iloc[-1]
    obs_tte_h = (t_end - t0).total_seconds() / 3600.0

    mc["obs_tte_h"] = obs_tte_h
    return mc


def plot_mc_histogram(mc: Dict, fig_path: Path) -> None:
    """Figure 5: Monte Carlo TTE histogram."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ttes = mc["tte_hours"]
    ax.hist(ttes, bins=30, color="steelblue", alpha=0.8, edgecolor="white")

    ax.axvline(mc["median"], color="red", linestyle="-", linewidth=2,
               label=f"Median: {mc['median']:.1f} h")
    ax.axvline(mc["ci_95_lo"], color="orange", linestyle="--", linewidth=1.5,
               label=f"95% CI: [{mc['ci_95_lo']:.1f}, {mc['ci_95_hi']:.1f}] h")
    ax.axvline(mc["ci_95_hi"], color="orange", linestyle="--", linewidth=1.5)

    if "obs_tte_h" in mc:
        ax.axvline(mc["obs_tte_h"], color="black", linestyle=":", linewidth=2,
                   label=f"Observed: {mc['obs_tte_h']:.1f} h")

    ax.set_xlabel("Time to Empty (hours)")
    ax.set_ylabel("Count")
    ax.set_title(f"Monte Carlo TTE (N={len(ttes)})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {fig_path}")


def run_tornado(
        session: pd.DataFrame,
        display_df: Optional[pd.DataFrame],
        params: bm.BatteryParams,
        cycle_count: int = 716
) -> List[Tuple[str, float, float, float]]:
    """Compute tornado sensitivity data."""
    usage_fn = build_usage_schedule(session, display_df, params)

    print("  Computing tornado sensitivity...")
    return bm.tornado_sensitivity(
        usage_fn, params, cycle_count=cycle_count, dt=60.0)


def plot_tornado(
        sensitivities: List[Tuple[str, float, float, float]],
        fig_path: Path
) -> None:
    """Figure 6: Tornado chart of parameter sensitivities."""
    fig, ax = plt.subplots(figsize=(10, 5))

    names = [s[0] for s in sensitivities]
    tte_nom = sensitivities[0][2]
    lo_vals = np.array([s[1] for s in sensitivities])
    hi_vals = np.array([s[3] for s in sensitivities])

    y_pos = np.arange(len(names))

    for idx in range(len(names)):
        lo = lo_vals[idx] - tte_nom
        hi = hi_vals[idx] - tte_nom
        left = min(lo, hi)
        width = abs(hi - lo)
        color = "steelblue" if hi > lo else "coral"
        ax.barh(y_pos[idx], width, left=left + tte_nom, height=0.6,
                color=color, alpha=0.8)

    ax.axvline(tte_nom, color="black", linestyle="-", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("TTE (hours)")
    ax.set_title(f"Tornado Sensitivity (Nominal: {tte_nom:.1f} h)")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {fig_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Multi-session validation (EPSQL -- 38 days, 5-min resolution)
#
# Two independent validation modes, each testing different model components:
#
#   1. CURRENT-DRIVEN (Coulomb counting test)
#      Input:  measured InstantAmperage from EPSQL
#      Tests:  Q_eff parameter, coulombic efficiency (eta), OCV curve
#      Output: SOC RMSE vs iOS-reported Level
#      Question answered: "Given the exact current draw, does the model
#      track SOC correctly?"
#
#   2. FORWARD SIMULATION (full-model TTE test)
#      Input:  power-calibrated usage schedule (V*I -> cpu_load)
#      Tests:  component power model, current solver, SOC dynamics, OCV
#      Output: TTE error (model vs observed drain rate extrapolation)
#      Question answered: "Given the phone's power consumption profile,
#      does the model predict battery life correctly?"
#
# The power-calibrated schedule uses measured V*I (battery discharge power)
# to set cpu_load so the model's total_power() matches reality. Screen state
# is inferred from current magnitude (|I| > 80mA = screen likely on).
# This is imperfect but much better than assuming screen always on.
# ─────────────────────────────────────────────────────────────────────────────

def build_power_calibrated_schedule(
        session: pd.DataFrame,
        params: bm.BatteryParams,
) -> bm.Callable[[float], Dict]:
    """Build usage schedule calibrated to match measured battery power.

    For each 5-min EPSQL interval:
      1. Compute measured power: P_meas = V * |I|  (from Voltage, InstantAmperage)
      2. Infer screen state: |I| > 80mA => screen on (heuristic)
      3. Set cpu_load so total_power(usage) ~= P_meas

    This preserves the model's power->current->SOC pipeline while ensuring
    the simulated power consumption matches what was actually measured.

    For sessions where display_df is available (PLSQL overlap window),
    use build_usage_schedule() instead -- it has real brightness data.
    """
    t0 = session["timestamp"].iloc[0]
    t_seconds = (session["timestamp"] - t0).dt.total_seconds().values

    voltage_v = pd.to_numeric(
        session["Voltage"], errors="coerce").fillna(3800).values / 1000.0
    current_ma = pd.to_numeric(
        session["InstantAmperage"], errors="coerce").fillna(0).values
    abs_current = np.abs(current_ma)

    # Measured battery discharge power (mW)
    measured_mw = voltage_v * abs_current

    # Heuristic screen state: >80mA suggests screen on, <30mA deep sleep
    screen_on = abs_current > 80.0

    # Calibrate cpu_load per interval so total_power ~= measured_mw
    # Cap at 1.0 for EPSQL: InstantAmperage is a point-in-time snapshot,
    # not an interval average, so current spikes overestimate 5-min power.
    # The clip at 1.0 compensates for this upward bias.
    n = len(session)
    cpu_loads = np.zeros(n)
    brightness = np.where(screen_on, 0.5, 0.0)

    for k in range(n):
        son = bool(screen_on[k])
        # Base power from all non-CPU-load components
        p_base = 0.0
        if son:
            p_base += params.screen_base_mw + params.screen_brightness_coeff_mw * 0.5
            p_base += params.overhead_screen_on_mw
        else:
            p_base += params.overhead_screen_off_mw
        p_base += params.wifi_idle_mw
        p_base += params.cellular_idle_mw
        p_base += params.bt_idle_mw

        if measured_mw[k] < params.cpu_idle_mw + 10:
            cpu_loads[k] = 0.0
        else:
            load = (measured_mw[k] - p_base) / params.cpu_load_coeff_mw
            cpu_loads[k] = float(np.clip(load, 0.0, 1.0))

    def schedule(t: float) -> Dict:
        idx = np.searchsorted(t_seconds, t, side="right") - 1
        idx = max(0, min(idx, n - 1))
        son = bool(screen_on[idx])
        return {
            "screen_on": son,
            "brightness": float(brightness[idx]),
            "cpu_load": float(cpu_loads[idx]),
            "wifi_state": "idle",
            "wifi_data_rate": 0.0,
            "cellular_state": "idle",
            "cellular_signal": 0.8,
            "cellular_data_rate": 0.0,
            "gps_state": "off",
            "bluetooth_state": "idle",
            "audio_state": "off",
            "camera_state": "off",
        }

    return schedule


def _session_temperature(session: pd.DataFrame) -> float:
    """Extract initial temperature from session, with sanity checks."""
    if "Temperature" not in session.columns:
        return 25.0
    temps = pd.to_numeric(session["Temperature"], errors="coerce").dropna()
    if len(temps) == 0:
        return 25.0
    T = float(temps.iloc[0])
    # EPSQL stores temperature * 100 in some firmware versions
    if T > 80:
        T /= 100.0
    return np.clip(T, -10.0, 50.0)


def validate_multi_session(
        epsql_df: pd.DataFrame,
        params: bm.BatteryParams,
        min_soc_drop: float = 10.0,
        display_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Validate model against all EPSQL discharge sessions.

    Runs two independent tests per session:

    1. Current-driven (tests Q_eff and OCV curve):
       Feeds measured InstantAmperage into model, compares model SOC to
       iOS Level. Low RMSE means capacity and efficiency params are correct.

    2. Forward simulation (tests full TTE pipeline):
       Uses power-calibrated schedule (V*I -> cpu_load), runs model from
       observed start SOC, compares drain rate. Low TTE error means the
       power->current->SOC pipeline works end-to-end.

    If display_df is provided, sessions overlapping PLSQL time window use
    real brightness data instead of the current-magnitude heuristic.

    Returns DataFrame with columns:
        session_idx, start_soc, end_soc, duration_h, T_init,
        cd_rmse_pct     - Current-driven SOC RMSE (% points)
        cd_final_err    - Current-driven final SOC error (% points)
        fwd_rmse_pct    - Forward sim SOC RMSE (% points)
        obs_tte_h       - Observed TTE (extrapolated from drain rate)
        model_tte_h     - Forward sim TTE (extrapolated from drain rate)
        used_display    - Whether PLSQL display data was available
    """
    sessions = extract_discharge_sessions(epsql_df, min_soc_drop)
    if not sessions:
        return pd.DataFrame()

    # PLSQL display time window (for overlap detection)
    plsql_t0 = None
    plsql_t1 = None
    if display_df is not None and len(display_df) > 0:
        plsql_t0 = display_df["timestamp"].iloc[0]
        plsql_t1 = display_df["timestamp"].iloc[-1]

    records: List[Dict] = []
    for idx, session in enumerate(sessions):
        t0 = session["timestamp"].iloc[0]
        duration_s = (session["timestamp"].iloc[-1] - t0).total_seconds()
        duration_h = duration_s / 3600.0
        soc_start = float(session["Level"].iloc[0]) / 100.0
        soc_end = float(session["Level"].iloc[-1]) / 100.0
        soc_drop = soc_start - soc_end

        if soc_drop <= 0 or duration_s < 600:
            continue

        T_init = _session_temperature(session)
        obs_t = (session["timestamp"] - t0).dt.total_seconds().values
        obs_soc = session["Level"].values.astype(float) / 100.0

        # --- Test 1: Current-driven (Coulomb counting) ---
        # Uses measured current directly. Tests Q_eff and eta only.
        raw_ma = pd.to_numeric(
            session["InstantAmperage"], errors="coerce").fillna(0).values
        current_a = -raw_ma / 1000.0  # iOS negative=discharge -> positive

        def _current_fn(t, _t_s=obs_t, _i=current_a):
            k = np.searchsorted(_t_s, t, side="right") - 1
            k = max(0, min(k, len(_i) - 1))
            return float(_i[k])

        try:
            cd_result = bm.simulate_current_driven(
                _current_fn, params,
                soc_init=soc_start,
                T_ambient=T_init, T_cell_init=T_init,
                t_max=duration_s, dt=300.0,
            )
            cd_soc_interp = np.interp(obs_t, cd_result["t"], cd_result["soc"])
            cd_rmse = float(np.sqrt(np.mean(
                (cd_soc_interp - obs_soc) ** 2)) * 100)
            cd_final_err = float(
                (cd_result["soc"][-1] - soc_end) * 100)
        except Exception as e:
            print(f"    Session {idx}: current-driven failed: {e}")
            cd_rmse = float("nan")
            cd_final_err = float("nan")

        # --- Test 2: Forward simulation (full model pipeline) ---
        # Check if this session overlaps PLSQL display data
        used_display = False
        if (plsql_t0 is not None and
                t0 >= plsql_t0 and
                session["timestamp"].iloc[-1] <= plsql_t1):
            usage_fn = build_usage_schedule(session, display_df, params)
            used_display = True
        else:
            usage_fn = build_power_calibrated_schedule(session, params)

        try:
            fwd_result = bm.simulate(
                usage_fn, params,
                soc_init=soc_start,
                T_ambient=T_init, T_cell_init=T_init,
                t_max=duration_s, dt=300.0,
            )
            fwd_soc_interp = np.interp(
                obs_t, fwd_result["t"], fwd_result["soc"])
            fwd_rmse = float(np.sqrt(np.mean(
                (fwd_soc_interp - obs_soc) ** 2)) * 100)

            # Extrapolate TTE from drain rates (100% -> 1%)
            obs_drain_rate = soc_drop / duration_h
            obs_tte_h = 0.99 / obs_drain_rate

            model_soc_drop = soc_start - float(fwd_result["soc"][-1])
            if model_soc_drop > 0.001:
                model_tte_h = 0.99 / (model_soc_drop / duration_h)
            else:
                model_tte_h = float("inf")
        except Exception as e:
            print(f"    Session {idx}: forward sim failed: {e}")
            fwd_rmse = float("nan")
            obs_tte_h = float("nan")
            model_tte_h = float("nan")

        records.append({
            "session_idx": idx,
            "start_soc": soc_start * 100,
            "end_soc": soc_end * 100,
            "soc_drop": soc_drop * 100,
            "duration_h": duration_h,
            "T_init": T_init,
            "cd_rmse_pct": cd_rmse,
            "cd_final_err": cd_final_err,
            "fwd_rmse_pct": fwd_rmse,
            "obs_tte_h": obs_tte_h,
            "model_tte_h": model_tte_h,
            "used_display": used_display,
        })

    return pd.DataFrame(records)


def plot_multi_session_scatter(
        results_df: pd.DataFrame,
        fig_path: Path
) -> None:
    """Figure 7: Model TTE vs Observed TTE scatter across sessions."""
    fig, ax = plt.subplots(figsize=(7, 7))

    obs = results_df["obs_tte_h"].values
    model = results_df["model_tte_h"].values

    # Filter unreasonable values
    mask = np.isfinite(obs) & np.isfinite(model) & (obs < 50) & (model < 50)
    obs, model = obs[mask], model[mask]

    ax.scatter(obs, model, c="steelblue", alpha=0.7, s=40,
               edgecolors="white", linewidth=0.5)

    # Identity line and +/-20% band
    lim = max(obs.max(), model.max()) * 1.1 if len(obs) > 0 else 30
    ax.plot([0, lim], [0, lim], "k--", alpha=0.5, label="Perfect prediction")
    x_line = np.linspace(0, lim, 100)
    ax.fill_between(x_line, x_line * 0.8, x_line * 1.2,
                    alpha=0.1, color="gray", label="+/- 20%")

    # Statistics annotation
    if len(obs) > 1:
        errors = model - obs
        mae = np.mean(np.abs(errors))
        mape = np.mean(np.abs(errors / obs)) * 100
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((obs - obs.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        ax.text(0.05, 0.95,
                f"N = {len(obs)} sessions\n"
                f"MAE = {mae:.1f} h\n"
                f"MAPE = {mape:.0f}%\n"
                f"R^2 = {r2:.3f}",
                transform=ax.transAxes, va="top", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_xlabel("Observed TTE (hours)")
    ax.set_ylabel("Model TTE (hours)")
    ax.set_title("Model vs Observed TTE (Multi-Session)")
    ax.legend(loc="lower right")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {fig_path}")


def plot_multi_session_errors(
        results_df: pd.DataFrame,
        fig_path: Path
) -> None:
    """Figure 8: Current-driven SOC RMSE, forward SOC RMSE, and TTE errors.

    Three panels showing what each validation mode reveals:
      Left:   Current-driven SOC RMSE -> tests Q_eff and OCV accuracy
      Center: Forward sim SOC RMSE    -> tests full power pipeline
      Right:  TTE prediction error    -> practical prediction accuracy
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Left: Current-driven SOC RMSE (tests capacity + OCV)
    cd_rmse = results_df["cd_rmse_pct"].dropna().values
    if len(cd_rmse) > 0:
        ax1.hist(cd_rmse, bins=15, color="steelblue", alpha=0.8,
                 edgecolor="white")
        ax1.axvline(np.median(cd_rmse), color="red", linestyle="--",
                    label=f"Median: {np.median(cd_rmse):.1f}%")
        ax1.legend()
    ax1.set_xlabel("SOC RMSE (% points)")
    ax1.set_ylabel("Count")
    ax1.set_title("Current-Driven\n(tests Q_eff, OCV)")
    ax1.grid(True, alpha=0.3)

    # Center: Forward sim SOC RMSE (tests full pipeline)
    fwd_rmse = results_df["fwd_rmse_pct"].dropna().values
    if len(fwd_rmse) > 0:
        ax2.hist(fwd_rmse, bins=15, color="teal", alpha=0.8,
                 edgecolor="white")
        ax2.axvline(np.median(fwd_rmse), color="red", linestyle="--",
                    label=f"Median: {np.median(fwd_rmse):.1f}%")
        ax2.legend()
    ax2.set_xlabel("SOC RMSE (% points)")
    ax2.set_ylabel("Count")
    ax2.set_title("Forward Sim\n(tests power pipeline)")
    ax2.grid(True, alpha=0.3)

    # Right: TTE error histogram
    mask = (np.isfinite(results_df["obs_tte_h"]) &
            np.isfinite(results_df["model_tte_h"]))
    tte_err = (results_df.loc[mask, "model_tte_h"] -
               results_df.loc[mask, "obs_tte_h"]).values
    tte_err = tte_err[np.abs(tte_err) < 20]

    if len(tte_err) > 0:
        ax3.hist(tte_err, bins=15, color="coral", alpha=0.8,
                 edgecolor="white")
        ax3.axvline(0, color="black", linestyle="-", linewidth=1)
        ax3.axvline(np.median(tte_err), color="red", linestyle="--",
                    label=f"Median: {np.median(tte_err):+.1f} h")
        ax3.legend()
    ax3.set_xlabel("TTE Error (model - obs, hours)")
    ax3.set_ylabel("Count")
    ax3.set_title("TTE Prediction Error\n(practical accuracy)")
    ax3.grid(True, alpha=0.3)

    fig.suptitle(f"Multi-Session Validation (N={len(results_df)} sessions)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fig_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(db_path: Path, figures_dir: Path,
                 n_mc: int = 500) -> None:
    """Run the full sysdiagnose analysis pipeline."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Opening powerlog: {db_path}")
    conn = open_powerlog(db_path)
    tables = list_tables(conn)
    print(f"  Found {len(tables)} tables")

    # --- Query data ---
    print("\nQuerying battery data...")
    batt_df = query_battery_table(conn)
    print(f"  {len(batt_df)} battery rows")
    print(f"  Time range: {batt_df['timestamp'].iloc[0]} to "
          f"{batt_df['timestamp'].iloc[-1]}")

    if "CycleCount" in batt_df.columns:
        cc_vals = pd.to_numeric(batt_df["CycleCount"], errors="coerce").dropna()
        cycle_count = int(cc_vals.iloc[-1]) if len(cc_vals) > 0 else 0
        print(f"  Cycle count: {cycle_count}")
    else:
        cycle_count = 0

    # Use NominalChargeCapacity from iOS as Q_ref (current effective capacity).
    # The paper's Eq 5 Q_ref should reflect the battery's actual health,
    # not the factory design capacity, since aging is not in the paper.
    Q_ref_mah = 4329.0  # fallback: design capacity
    if "NominalChargeCapacity" in batt_df.columns:
        ncc = pd.to_numeric(
            batt_df["NominalChargeCapacity"], errors="coerce").dropna()
        if len(ncc) > 0:
            Q_ref_mah = float(ncc.iloc[-1])
            print(f"  NominalChargeCapacity: {Q_ref_mah:.0f} mAh "
                  f"({Q_ref_mah / 4329.0 * 100:.0f}% health)")

    params = bm.BatteryParams(Q_design_mah=Q_ref_mah)

    display_df = query_display_table(conn)
    if display_df is not None:
        print(f"  {len(display_df)} display rows")
    else:
        print("  No display data found")

    # --- 2b. Component energy breakdown (Figure 1) ---
    print("\n--- Component Energy Breakdown ---")
    node_df = query_component_energy(conn)
    breakdown = compute_component_breakdown(node_df)
    if breakdown is not None and len(breakdown) > 0:
        total_e = breakdown["Energy"].sum()
        for _, row in breakdown.iterrows():
            pct = row["Energy"] / total_e * 100
            print(f"  {row['Component']:15s}  {pct:5.1f}%")
        plot_component_energy(
            breakdown, figures_dir / "sysdiag_component_energy.png")
    else:
        print("  No component energy data available")

    # --- 2c. App category breakdown (Figure 2) ---
    print("\n--- App Category Energy ---")
    app_df = query_app_energy(conn)
    cat_df = compute_app_categories(app_df)
    if cat_df is not None and len(cat_df) > 0:
        total_e = cat_df["Energy"].sum()
        for _, row in cat_df.iterrows():
            pct = row["Energy"] / total_e * 100
            print(f"  {row['Category']:15s}  {pct:5.1f}%")
        plot_app_categories(
            cat_df, figures_dir / "sysdiag_app_categories.png")
    else:
        print("  No app energy data available")

    # --- 2d. Extract discharge sessions ---
    print("\n--- Discharge Sessions ---")
    sessions = extract_discharge_sessions(batt_df, min_soc_drop=20.0)
    print(f"  Found {len(sessions)} discharge sessions (>20% drop)")
    for idx, s in enumerate(sessions):
        soc0 = s["Level"].iloc[0]
        soc1 = s["Level"].iloc[-1]
        dur_h = (s["timestamp"].iloc[-1] - s["timestamp"].iloc[0]).total_seconds() / 3600
        print(f"    [{idx}] {soc0:.0f}% -> {soc1:.0f}%  ({dur_h:.1f} h)")

    session = find_main_session(sessions)
    if session is None:
        print("  No valid discharge session found. Skipping simulation steps.")
        conn.close()
        return

    soc0 = session["Level"].iloc[0]
    soc1 = session["Level"].iloc[-1]
    dur_h = (session["timestamp"].iloc[-1] - session["timestamp"].iloc[0]).total_seconds() / 3600
    print(f"  Main session: {soc0:.0f}% -> {soc1:.0f}% ({dur_h:.1f} h)")

    # --- 2e. Forward simulation vs observed SOC (Figure 3) ---
    print("\n--- Forward Simulation ---")
    fwd = forward_simulation(session, display_df, params, cycle_count)
    model_tte = bm.time_to_empty_hours(fwd["model"])
    print(f"  Model TTE: {model_tte:.1f} h")
    print(f"  Observed session duration: {dur_h:.1f} h")

    soc_model_final = fwd["model"]["soc"][-1]
    soc_obs_final = fwd["obs_soc"][-1]
    print(f"  Model final SOC: {soc_model_final*100:.1f}%")
    print(f"  Observed final SOC: {soc_obs_final*100:.1f}%")

    plot_soc_comparison(fwd, figures_dir / "sysdiag_soc_comparison.png")

    # --- 2f. Model vs observed voltage (Figure 4) ---
    print("\n--- Voltage Comparison ---")
    plot_voltage_comparison(fwd, figures_dir / "sysdiag_voltage_comparison.png")

    # --- 2g. Current-driven validation ---
    print("\n--- Current-Driven Validation ---")
    cd = current_driven_validation(session, params, cycle_count)
    if cd is not None:
        cd_soc_final = cd["model"]["soc"][-1]
        print(f"  Coulomb-counted final SOC: {cd_soc_final*100:.1f}%")
        print(f"  Observed final SOC: {soc_obs_final*100:.1f}%")
        soc_err = abs(cd_soc_final - soc_obs_final) * 100
        print(f"  SOC error: {soc_err:.1f}%")
    else:
        print("  No InstantAmperage data -- skipping")

    # --- 2h. Monte Carlo TTE (Figure 5) ---
    print(f"\n--- Monte Carlo TTE (N={n_mc}) ---")
    mc = run_monte_carlo(session, display_df, params, cycle_count, n_mc)
    print(f"  Median: {mc['median']:.1f} h")
    print(f"  Mean:   {mc['mean']:.1f} h (+/- {mc['std']:.1f})")
    print(f"  95% CI: [{mc['ci_95_lo']:.1f}, {mc['ci_95_hi']:.1f}] h")
    print(f"  Observed: {mc['obs_tte_h']:.1f} h")
    brackets = mc["ci_95_lo"] <= mc["obs_tte_h"] <= mc["ci_95_hi"]
    print(f"  95% CI brackets observed: {'YES' if brackets else 'NO'}")

    plot_mc_histogram(mc, figures_dir / "sysdiag_mc_tte.png")

    # --- Tornado (Figure 6) ---
    print("\n--- Tornado Sensitivity ---")
    tornado = run_tornado(session, display_df, params, cycle_count)
    for name, lo, nom, hi in tornado:
        print(f"  {name:30s}  [{lo:.1f}, {hi:.1f}]  "
              f"swing={abs(hi-lo):.1f} h")
    plot_tornado(tornado, figures_dir / "sysdiag_tornado.png")

    # --- EPSQL multi-session validation (Figures 7-8) ---
    #
    # Two independent tests per session:
    #   Current-driven: measured I(t) -> model SOC vs iOS Level
    #     Tests Q_eff (capacity) and OCV curve accuracy
    #   Forward sim: power-calibrated schedule -> model SOC vs iOS Level
    #     Tests full pipeline: component power -> current -> SOC -> TTE
    #
    # Sessions overlapping PLSQL window use real display brightness.
    # Other sessions use V*I power calibration with screen heuristic.
    print("\n--- EPSQL Multi-Session Validation ---")
    epsql_path = find_epsql(db_path)
    if epsql_path is not None:
        print(f"  Found EPSQL: {epsql_path.name}")
        epsql_df = query_epsql_battery(epsql_path)
        if epsql_df is not None:
            print(f"  {len(epsql_df)} EPSQL rows, 5-min resolution")
            print(f"  Time range: {epsql_df['timestamp'].iloc[0]} to "
                  f"{epsql_df['timestamp'].iloc[-1]}")

            epsql_sessions = extract_discharge_sessions(
                epsql_df, min_soc_drop=10.0)
            print(f"  Found {len(epsql_sessions)} discharge sessions "
                  f"(>10% drop)")

            if epsql_sessions:
                print("  Running dual-mode validation...")
                ms_results = validate_multi_session(
                    epsql_df, params, min_soc_drop=10.0,
                    display_df=display_df)

                if len(ms_results) > 0:
                    n_disp = ms_results["used_display"].sum()
                    n_pcal = len(ms_results) - n_disp
                    print(f"  Validated {len(ms_results)} sessions "
                          f"({n_disp} with display data, "
                          f"{n_pcal} power-calibrated)")

                    # Current-driven results (capacity/OCV test)
                    cd = ms_results["cd_rmse_pct"].dropna()
                    if len(cd) > 0:
                        print(f"\n  Current-driven (tests Q_eff + OCV):")
                        print(f"    SOC RMSE: median {cd.median():.1f}%, "
                              f"mean {cd.mean():.1f}%")
                        cd_err = ms_results["cd_final_err"].dropna()
                        print(f"    Final SOC error: median "
                              f"{cd_err.median():+.1f}%, "
                              f"mean {cd_err.mean():+.1f}%")

                    # Forward sim results (full pipeline test)
                    fwd = ms_results["fwd_rmse_pct"].dropna()
                    if len(fwd) > 0:
                        print(f"\n  Forward sim (tests full pipeline):")
                        print(f"    SOC RMSE: median {fwd.median():.1f}%, "
                              f"mean {fwd.mean():.1f}%")

                    valid_tte = ms_results[
                        np.isfinite(ms_results["obs_tte_h"]) &
                        np.isfinite(ms_results["model_tte_h"])
                    ]
                    if len(valid_tte) > 0:
                        tte_err = (valid_tte["model_tte_h"] -
                                   valid_tte["obs_tte_h"])
                        print(f"\n  TTE prediction:")
                        print(f"    MAE: {np.abs(tte_err).mean():.1f} h")
                        mape = np.mean(
                            np.abs(tte_err / valid_tte["obs_tte_h"])) * 100
                        print(f"    MAPE: {mape:.0f}%")
                        print(f"    Median bias: "
                              f"{tte_err.median():+.1f} h")

                    plot_multi_session_scatter(
                        ms_results,
                        figures_dir / "sysdiag_multi_session_tte.png")
                    plot_multi_session_errors(
                        ms_results,
                        figures_dir / "sysdiag_multi_session_errors.png")
                else:
                    print("  No sessions validated successfully")
            else:
                print("  No discharge sessions found in EPSQL")
        else:
            print("  Could not read EPSQL battery data")
    else:
        print("  No EPSQL database found (skipping multi-session)")

    conn.close()
    print("\nPipeline complete.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sysdiagnose-based battery TTE analysis pipeline")
    parser.add_argument(
        "--db", type=Path, default=DEFAULT_DB,
        help=f"Path to CurrentPowerlog.PLSQL (default: {DEFAULT_DB})")
    parser.add_argument(
        "--figures-dir", type=Path, default=DEFAULT_FIGURES,
        help=f"Output directory for figures (default: {DEFAULT_FIGURES})")
    parser.add_argument(
        "--mc-samples", type=int, default=500,
        help="Number of Monte Carlo samples (default: 500)")
    args = parser.parse_args()

    run_pipeline(args.db, args.figures_dir, args.mc_samples)


if __name__ == "__main__":
    main()
