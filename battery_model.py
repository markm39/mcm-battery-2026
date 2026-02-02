"""
battery_model.py -- Paper-exact smartphone battery ODE model (self-contained).

Implements the Enhanced Coulomb Counting (ECC) model with N-RC equivalent
circuit, linear temperature dependence, and component-level power modeling.

Paper equations (numbered as in manuscript):

  Eq 1: C_th * dT/dt = Q_gen - hA * (T - T_env)
  Eq 2: Q_gen = i^2 * R0(z,T) + sum_j( i_Rj^2 * Rj(T) )
  Eq 3: R0(T) = R0_ref * (1 + alpha_R0 * max(T_ref - T, 0))
  Eq 4: Rj(T) = Rj_ref * (1 + alpha_Rj * max(T_ref - T, 0))
  Eq 5: Q_eff(T) = Q_ref * max(1 - alpha_Q * max(T_ref - T, 0), 0)
  Eq 6: dz/dt = -eta * i / (Q_eff_Ah * 3600)

  Sec 4.2.2: di_Rj/dt = (i - i_Rj) / (Rj(T) * Cj)
  Sec 4.2.3: dh/dt = -gamma * |dz/dt| * (h + M0 * sgn(i))
  Sec 4.2.4: v = OCV(z) - R0(z,T)*i - sum_j(Rj(T)*i_Rj) + h + M1*sgn(i)

State vector: [z, T, i_R1, ..., i_RN, h]   (N_RC + 3 elements)

Note: The paper does NOT include aging/cycle degradation. Aging is available
as an optional extension (alpha_fade parameter) but defaults to 0 (disabled).

Dependencies: numpy, scipy (only).
No imports from the model/ package -- fully self-contained.
"""

from __future__ import annotations

import numpy as np
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
from scipy.integrate import solve_ivp


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Lookup Tables
# ═══════════════════════════════════════════════════════════════════════════════

# 21-point OCV table from iPhone sysdiagnose near-rest measurements.
# Source: PLBatteryAgent_EventBackward_Battery rows with |InstantAmperage| < 50 mA
# (IR drop < 5mV at typical resistance), supplemented by EPSQL BDC_SBC near-rest
# for SOC points with sparse high-res data. SOC 0% extrapolated to 3.30V
# (min observed voltage under load was 3.20V).
OCV_SOC = np.array([
    0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00,
])
OCV_VOLTAGE = np.array([
    3.300, 3.653, 3.670, 3.700, 3.731, 3.750, 3.770, 3.789, 3.808, 3.833,
    3.858, 3.895, 3.931, 3.980, 4.030, 4.092, 4.154, 4.198, 4.243, 4.310,
    4.376,
])

# 15-point SOC-dependent resistance table (Ohms)
# Source: iPhone iOS Analytics RaTable (716 cycles, 86% health)
RA_SOC = np.array([
    0.000, 0.071, 0.143, 0.214, 0.286, 0.357, 0.429, 0.500,
    0.571, 0.643, 0.714, 0.786, 0.857, 0.929, 1.000,
])
RA_OHM = np.array([
    134, 169, 173, 183, 190, 129, 145, 140,
    157, 157, 175, 165, 212, 521, 1174,
], dtype=float) / 1000.0


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Parameter Dataclasses
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RCBranch:
    """Single RC polarization branch (Sec 4.2.2).

    Captures voltage relaxation after load transients.
    tau = R_ref * C is the time constant in seconds.
    """
    R_ref: float            # Resistance at T_ref (Ohms)
    C: float                # Capacitance (Farads)
    alpha_R: float = 0.008  # Linear temp coefficient for this branch (/C)

    @property
    def tau(self) -> float:
        return self.R_ref * self.C


@dataclass
class BatteryParams:
    """Complete battery model parameters.

    Defaults calibrated against iPhone iOS analytics (716 cycles, 86% health)
    and NASA PCoE B0005 discharge data.
    """
    # --- Cell electrochemistry ---
    Q_design_mah: float = 4329.0     # Design capacity (mAh)
    R0_ref: float = 0.159            # Nominal series resistance at T_ref (Ohms)
    eta_discharge: float = 1.0       # Coulombic efficiency, discharge [Eq 6]
    eta_charge: float = 0.99         # Coulombic efficiency, charge

    # --- RC branches (default: N=1, tau=60s) ---
    rc_branches: List[RCBranch] = field(default_factory=lambda: [
        RCBranch(R_ref=0.050, C=1200.0, alpha_R=0.008),
    ])

    # --- Hysteresis (Sec 4.2.3) ---
    gamma: float = 1.0               # Rate constant
    M0: float = 0.005                # Max hysteresis voltage (V)
    M1: float = 0.005                # Instantaneous hysteresis (V)

    # --- Thermal (Eqs 1, 3-5) ---
    C_th: float = 40.0               # Thermal mass (J/C)
    hA: float = 0.5                  # Heat dissipation coefficient (W/C)
    alpha_R0: float = 0.008          # R0 linear temp coeff (/C below T_ref)
    alpha_Q: float = 0.005           # Capacity temp coeff (/C below T_ref)
    T_ref: float = 25.0              # Reference temperature (C)

    # --- Aging (NOT in paper; optional extension, off by default) ---
    # Set alpha_fade > 0 to enable cycle-dependent capacity fade.
    # E.g. alpha_fade=0.005232, beta_fade=0.5 gives 86% at 716 cycles.
    alpha_fade: float = 0.0          # Power-law fade coefficient (0 = disabled)
    beta_fade: float = 0.5           # Exponent (0.5 = sqrt = SEI diffusion)
    calendar_fade_per_day: float = 0.0
    R_growth_per_cycle: float = 0.0

    # --- Component power (mW) ---
    screen_base_mw: float = 50.0
    screen_brightness_coeff_mw: float = 400.0
    cpu_idle_mw: float = 5.0
    cpu_load_coeff_mw: float = 1500.0
    # Two-state system overhead calibrated from PLSQL screen-on/off power:
    #   Screen ON  median V*|I| = 506 mW - 250 (display) - 25 (radio) = 231 mW
    #   Screen OFF median V*|I| = 263 mW - 25 (radio) = 238 mW
    # Source: PLBatteryAgent + PLDisplayAgent cross-reference, 4939 discharge rows
    overhead_screen_on_mw: float = 230.0
    overhead_screen_off_mw: float = 240.0
    wifi_idle_mw: float = 10.0
    wifi_active_mw: float = 800.0
    wifi_scanning_mw: float = 300.0
    cellular_idle_mw: float = 10.0
    cellular_active_mw: float = 1200.0
    cellular_poor_signal_mult: float = 1.5
    gps_active_mw: float = 176.0
    gps_background_mw: float = 30.0
    bt_idle_mw: float = 5.0
    bt_active_mw: float = 50.0
    audio_speaker_mw: float = 200.0
    audio_headphone_mw: float = 20.0
    camera_viewfinder_mw: float = 1000.0
    camera_recording_mw: float = 1500.0

    # --- Simulation defaults ---
    soc_empty: float = 0.01
    t_max_s: float = 86400.0
    T_ambient: float = 25.0

    @property
    def N_RC(self) -> int:
        return len(self.rc_branches)

    @property
    def state_dim(self) -> int:
        """Length of state vector [z, T, i_R1..i_RN, h]."""
        return self.N_RC + 3


# Module-level default parameter set
DEFAULT = BatteryParams()


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: OCV and Resistance Lookup
# ═══════════════════════════════════════════════════════════════════════════════

def ocv(z: float) -> float:
    """Open-circuit voltage from 21-point lookup table with linear interp.

    Clamped to [3.0, 4.45] V (iPhone NMC/LCO cell range).
    """
    z_c = np.clip(z, 0.0, 1.0)
    return float(np.clip(np.interp(z_c, OCV_SOC, OCV_VOLTAGE), 3.0, 4.45))


def resistance_at_soc(z: float) -> float:
    """SOC-dependent internal resistance (Ohms) from 15-point RA table.

    U-shaped: high at low SOC, very high near full charge (CV region).
    """
    z_c = np.clip(z, 0.0, 1.0)
    return float(np.interp(z_c, RA_SOC, RA_OHM))


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: Thermal Functions (Paper Eqs 3-5)
# ═══════════════════════════════════════════════════════════════════════════════

def R0_at_temp(R0_ref: float, T: float,
               alpha_R0: float = 0.008, T_ref: float = 25.0) -> float:
    """Series resistance adjusted for temperature [Eq 3].

    R0(T) = R0_ref * (1 + alpha_R0 * max(T_ref - T, 0))
    Linear increase below T_ref; unchanged above.
    """
    return R0_ref * (1.0 + alpha_R0 * max(T_ref - T, 0.0))


def Rj_at_temp(Rj_ref: float, T: float,
               alpha_Rj: float = 0.008, T_ref: float = 25.0) -> float:
    """RC branch resistance adjusted for temperature [Eq 4].

    Rj(T) = Rj_ref * (1 + alpha_Rj * max(T_ref - T, 0))
    """
    return Rj_ref * (1.0 + alpha_Rj * max(T_ref - T, 0.0))


def Q_eff_at_temp(Q_nom: float, T: float,
                  alpha_Q: float = 0.005, T_ref: float = 25.0) -> float:
    """Effective capacity adjusted for temperature [Eq 5].

    Q_eff(T) = Q_nom * max(1 - alpha_Q * max(T_ref - T, 0), 0)

    Double-max clipping: inner max prevents cold-boost above T_ref;
    outer max prevents capacity going negative at extreme cold.
    """
    return Q_nom * max(1.0 - alpha_Q * max(T_ref - T, 0.0), 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: Aging Functions (optional extension -- not in paper)
# ═══════════════════════════════════════════════════════════════════════════════

def capacity_after_cycles(n_cycles: int,
                          params: BatteryParams = DEFAULT) -> float:
    """Battery capacity after n charge cycles (mAh).

    Power-law fade: Q(n) = Q_design * max(1 - alpha * n^beta, 0.5)
    Returns Q_design unchanged when alpha_fade == 0 (paper-default).
    """
    if params.alpha_fade <= 0.0:
        return params.Q_design_mah
    fade = params.alpha_fade * (max(n_cycles, 0) ** params.beta_fade)
    return params.Q_design_mah * max(1.0 - fade, 0.5)


def effective_capacity(cycle_count: int, T: float,
                       params: BatteryParams = DEFAULT) -> float:
    """Effective capacity combining optional aging and temperature [Eq 5].

    When alpha_fade == 0 (default), this reduces to the paper's Eq 5:
        Q_eff(T) = Q_design * max(1 - alpha_Q * max(T_ref - T, 0), 0)
    """
    Q_base = capacity_after_cycles(cycle_count, params)
    return Q_eff_at_temp(Q_base, T, params.alpha_Q, params.T_ref)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6: Component Power Functions (all return mW)
# ═══════════════════════════════════════════════════════════════════════════════

def screen_power(brightness: float, screen_on: bool,
                 params: BatteryParams = DEFAULT) -> float:
    if not screen_on:
        return 0.0
    return params.screen_base_mw + params.screen_brightness_coeff_mw * brightness


def cpu_power(load: float, screen_on: bool,
              params: BatteryParams = DEFAULT) -> float:
    """CPU + system overhead power.

    Three regimes:
      Deep sleep (screen off, load < 1%):  cpu_idle_mw (~5 mW)
      Screen off, active:  overhead_screen_off_mw + load * cpu_load_coeff_mw
      Screen on:           overhead_screen_on_mw  + load * cpu_load_coeff_mw
    """
    if not screen_on and load < 0.01:
        return params.cpu_idle_mw
    overhead = (params.overhead_screen_on_mw if screen_on
                else params.overhead_screen_off_mw)
    return overhead + params.cpu_load_coeff_mw * load


def wifi_power(state: str, data_rate: float = 0.0,
               params: BatteryParams = DEFAULT) -> float:
    if state == "off":
        return 0.0
    if state == "scanning":
        return params.wifi_scanning_mw
    if state == "active":
        return params.wifi_idle_mw + params.wifi_active_mw * data_rate
    return params.wifi_idle_mw  # "idle"


def cellular_power(state: str, signal: float = 1.0,
                   data_rate: float = 0.0,
                   params: BatteryParams = DEFAULT) -> float:
    if state == "off":
        return 0.0
    penalty = 1.0 + (1.0 - signal) * (params.cellular_poor_signal_mult - 1.0)
    if state == "active":
        base = params.cellular_idle_mw + params.cellular_active_mw * data_rate
        return base * penalty
    return params.cellular_idle_mw * penalty  # "idle"


def gps_power(state: str, params: BatteryParams = DEFAULT) -> float:
    if state == "active":
        return params.gps_active_mw
    if state == "background":
        return params.gps_background_mw
    return 0.0


def bluetooth_power(state: str, params: BatteryParams = DEFAULT) -> float:
    if state == "active":
        return params.bt_active_mw
    if state == "idle":
        return params.bt_idle_mw
    return 0.0


def audio_power(state: str, params: BatteryParams = DEFAULT) -> float:
    if state == "speaker":
        return params.audio_speaker_mw
    if state == "headphone":
        return params.audio_headphone_mw
    return 0.0


def camera_power(state: str, params: BatteryParams = DEFAULT) -> float:
    if state == "viewfinder":
        return params.camera_viewfinder_mw
    if state == "recording":
        return params.camera_recording_mw
    return 0.0


def total_power(usage: Dict, params: BatteryParams = DEFAULT) -> float:
    """Sum all component power draws (mW) from a usage state dict."""
    p = 0.0
    son = usage.get("screen_on", True)
    p += screen_power(usage.get("brightness", 0.5), son, params)
    p += cpu_power(usage.get("cpu_load", 0.1), son, params)
    p += wifi_power(usage.get("wifi_state", "idle"),
                    usage.get("wifi_data_rate", 0.0), params)
    p += cellular_power(usage.get("cellular_state", "idle"),
                        usage.get("cellular_signal", 0.8),
                        usage.get("cellular_data_rate", 0.0), params)
    p += gps_power(usage.get("gps_state", "off"), params)
    p += bluetooth_power(usage.get("bluetooth_state", "off"), params)
    p += audio_power(usage.get("audio_state", "off"), params)
    p += camera_power(usage.get("camera_state", "off"), params)
    return p


# ═══════════════════════════════════════════════════════════════════════════════
# Section 7: Terminal Voltage and Current Solver
# ═══════════════════════════════════════════════════════════════════════════════

def terminal_voltage(z: float, i: float, T: float,
                     i_R: List[float], h: float,
                     params: BatteryParams = DEFAULT) -> float:
    """Battery terminal voltage under load [Sec 4.2.4].

    v = OCV(z) - R0(z,T)*i - sum_j(Rj(T)*i_Rj) + h + M1*sgn(i)
    """
    R0_soc = resistance_at_soc(z)
    R0 = R0_at_temp(R0_soc, T, params.alpha_R0, params.T_ref)

    v = ocv(z) - R0 * i

    for j, branch in enumerate(params.rc_branches):
        Rj = Rj_at_temp(branch.R_ref, T, branch.alpha_R, params.T_ref)
        v -= Rj * i_R[j]

    v += h
    if i > 0:
        v += params.M1
    elif i < 0:
        v -= params.M1

    return v


def solve_current(P_load_w: float, z: float, T: float,
                  i_R: List[float], h: float,
                  params: BatteryParams = DEFAULT) -> float:
    """Solve for discharge current given load power (quadratic, exact).

    From P = v*i and v = V_eff - R0*i, the power balance gives:
        R0*i^2 - V_eff*i + P = 0
        i = (V_eff - sqrt(V_eff^2 - 4*R0*P)) / (2*R0)

    The smaller root is the physical solution (higher terminal voltage).
    """
    if P_load_w <= 0.0:
        return 0.0

    R0_soc = resistance_at_soc(z)
    R0 = R0_at_temp(R0_soc, T, params.alpha_R0, params.T_ref)

    # Effective OCV including RC and hysteresis (sgn(i)=+1 for discharge)
    V_eff = ocv(z)
    for j, branch in enumerate(params.rc_branches):
        Rj = Rj_at_temp(branch.R_ref, T, branch.alpha_R, params.T_ref)
        V_eff -= Rj * i_R[j]
    V_eff += h + params.M1

    if V_eff <= 0.1:
        V_eff = 3.0  # safety floor

    disc = V_eff * V_eff - 4.0 * R0 * P_load_w
    if disc < 0.0:
        # Power exceeds battery capability; return max-power current
        return V_eff / (2.0 * R0)

    return (V_eff - np.sqrt(disc)) / (2.0 * R0)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 8: Core ODE System (Paper Eqs 1-6, Sec 4.2.2-4.2.4)
# ═══════════════════════════════════════════════════════════════════════════════

def battery_rhs(t: float, state: np.ndarray,
                usage_fn: Callable[[float], Dict],
                params: BatteryParams = DEFAULT,
                T_env: float = 25.0,
                cycle_count: int = 0) -> np.ndarray:
    """RHS of the battery ODE (forward / power-driven mode).

    State: [z, T, i_R1, ..., i_RN, h]

    Current i is solved from the power constraint P_load = v * i.
    Heat generation uses battery-internal I^2*R losses only [Eq 2].
    """
    N = params.N_RC

    z = np.clip(state[0], 0.0, 1.0)
    T = state[1]
    i_R = list(state[2:2 + N]) if N > 0 else []
    h = state[2 + N]

    # Component power
    usage = usage_fn(t)
    P_mw = total_power(usage, params)
    P_w = P_mw / 1000.0

    # Effective capacity (aging + temperature)
    Q_eff = effective_capacity(cycle_count, T, params)
    Q_eff_Ah = Q_eff / 1000.0

    # Solve current from power constraint
    i = solve_current(P_w, z, T, i_R, h, params)
    eta = params.eta_discharge if i >= 0 else params.eta_charge

    # --- Eq 6: SOC dynamics ---
    dz_dt = -eta * i / (Q_eff_Ah * 3600.0)

    # --- Eq 2: Heat generation (battery internal losses only) ---
    R0_soc = resistance_at_soc(z)
    R0 = R0_at_temp(R0_soc, T, params.alpha_R0, params.T_ref)
    Q_gen = i * i * R0
    for j, branch in enumerate(params.rc_branches):
        Rj = Rj_at_temp(branch.R_ref, T, branch.alpha_R, params.T_ref)
        Q_gen += i_R[j] * i_R[j] * Rj

    # --- Eq 1: Thermal dynamics ---
    dT_dt = (Q_gen - params.hA * (T - T_env)) / params.C_th

    # --- Sec 4.2.2: RC branch dynamics ---
    di_R = []
    for j, branch in enumerate(params.rc_branches):
        Rj = Rj_at_temp(branch.R_ref, T, branch.alpha_R, params.T_ref)
        tau_j = Rj * branch.C
        di_R.append((i - i_R[j]) / tau_j)

    # --- Sec 4.2.3: Hysteresis dynamics ---
    sgn_i = 1.0 if i > 0 else (-1.0 if i < 0 else 0.0)
    dh_dt = -params.gamma * abs(dz_dt) * (h + params.M0 * sgn_i)

    deriv = np.zeros(N + 3)
    deriv[0] = dz_dt
    deriv[1] = dT_dt
    for j in range(N):
        deriv[2 + j] = di_R[j]
    deriv[2 + N] = dh_dt

    return deriv


def battery_rhs_current_driven(t: float, state: np.ndarray,
                               current_fn: Callable[[float], float],
                               params: BatteryParams = DEFAULT,
                               T_env: float = 25.0,
                               cycle_count: int = 0) -> np.ndarray:
    """RHS for current-driven mode (validation against measured current).

    Same equations as battery_rhs, but i(t) is externally provided
    rather than solved from the power constraint.
    """
    N = params.N_RC

    z = np.clip(state[0], 0.0, 1.0)
    T = state[1]
    i_R = list(state[2:2 + N]) if N > 0 else []
    h = state[2 + N]

    i = current_fn(t)
    eta = params.eta_discharge if i >= 0 else params.eta_charge

    Q_eff = effective_capacity(cycle_count, T, params)
    Q_eff_Ah = Q_eff / 1000.0

    dz_dt = -eta * i / (Q_eff_Ah * 3600.0)

    R0_soc = resistance_at_soc(z)
    R0 = R0_at_temp(R0_soc, T, params.alpha_R0, params.T_ref)
    Q_gen = i * i * R0
    for j, branch in enumerate(params.rc_branches):
        Rj = Rj_at_temp(branch.R_ref, T, branch.alpha_R, params.T_ref)
        Q_gen += i_R[j] * i_R[j] * Rj

    dT_dt = (Q_gen - params.hA * (T - T_env)) / params.C_th

    di_R = []
    for j, branch in enumerate(params.rc_branches):
        Rj = Rj_at_temp(branch.R_ref, T, branch.alpha_R, params.T_ref)
        tau_j = Rj * branch.C
        di_R.append((i - i_R[j]) / tau_j)

    sgn_i = 1.0 if i > 0 else (-1.0 if i < 0 else 0.0)
    dh_dt = -params.gamma * abs(dz_dt) * (h + params.M0 * sgn_i)

    deriv = np.zeros(N + 3)
    deriv[0] = dz_dt
    deriv[1] = dT_dt
    for j in range(N):
        deriv[2 + j] = di_R[j]
    deriv[2 + N] = dh_dt

    return deriv


# ═══════════════════════════════════════════════════════════════════════════════
# Section 9: Simulation Runners
# ═══════════════════════════════════════════════════════════════════════════════

def simulate(usage_fn: Callable[[float], Dict],
             params: BatteryParams = DEFAULT,
             soc_init: float = 1.0,
             T_ambient: Optional[float] = None,
             T_cell_init: Optional[float] = None,
             cycle_count: int = 0,
             t_max: Optional[float] = None,
             dt: float = 60.0) -> Dict:
    """Run forward (power-driven) battery simulation.

    Args:
        usage_fn: Callable(t) -> usage_state dict
        params: Battery parameters
        soc_init: Initial SOC (0-1)
        T_ambient: Ambient temperature (C)
        T_cell_init: Initial cell temperature (C)
        cycle_count: Battery age in cycles
        t_max: Simulation end time (s)
        dt: Output time step (s)

    Returns:
        Dict with: t, soc, T_cell, voltage, current_ma, power_mw,
        time_to_empty, i_R (list of arrays), h, solver_status
    """
    if T_ambient is None:
        T_ambient = params.T_ambient
    if T_cell_init is None:
        T_cell_init = T_ambient
    if t_max is None:
        t_max = params.t_max_s

    N = params.N_RC
    y0 = np.zeros(N + 3)
    y0[0] = soc_init
    y0[1] = T_cell_init
    # i_R[j] and h start at 0

    t_eval = np.arange(0.0, t_max, dt)

    def soc_empty_event(t, y, *_args):
        return y[0] - params.soc_empty
    soc_empty_event.terminal = True
    soc_empty_event.direction = -1

    # Solver tuning: reduce max_step for fast RC dynamics
    min_tau = min((b.tau for b in params.rc_branches), default=1e6)
    max_step = min(60.0, min_tau / 2.0) if N > 0 else 60.0
    method = "Radau" if (N > 0 and min_tau < 20.0) else "RK45"

    sol = solve_ivp(
        fun=lambda t, y: battery_rhs(
            t, y, usage_fn, params, T_ambient, cycle_count),
        t_span=(0.0, t_max),
        y0=y0,
        method=method,
        t_eval=t_eval,
        events=soc_empty_event,
        max_step=max_step,
        rtol=1e-6,
        atol=1e-9,
    )

    t_out = sol.t
    soc_out = np.clip(sol.y[0], 0.0, 1.0)
    T_out = sol.y[1]
    i_R_out = [sol.y[2 + j] for j in range(N)]
    h_out = sol.y[2 + N]

    # Compute derived quantities at each output point
    n_pts = len(t_out)
    voltage = np.zeros(n_pts)
    current_ma = np.zeros(n_pts)
    power_mw = np.zeros(n_pts)

    for k in range(n_pts):
        usage = usage_fn(t_out[k])
        pw = total_power(usage, params)
        power_mw[k] = pw
        i_R_k = [i_R_out[j][k] for j in range(N)]
        i_a = solve_current(pw / 1000.0, soc_out[k], T_out[k],
                            i_R_k, h_out[k], params)
        current_ma[k] = i_a * 1000.0
        voltage[k] = terminal_voltage(soc_out[k], i_a, T_out[k],
                                      i_R_k, h_out[k], params)

    tte = None
    if sol.t_events and len(sol.t_events[0]) > 0:
        tte = float(sol.t_events[0][0])

    return {
        "t": t_out,
        "soc": soc_out,
        "T_cell": T_out,
        "voltage": voltage,
        "current_ma": current_ma,
        "power_mw": power_mw,
        "time_to_empty": tte,
        "i_R": i_R_out,
        "h": h_out,
        "solver_status": sol.status,
        "solver_message": sol.message,
    }


def simulate_current_driven(
        current_fn: Callable[[float], float],
        params: BatteryParams = DEFAULT,
        soc_init: float = 1.0,
        T_ambient: Optional[float] = None,
        T_cell_init: Optional[float] = None,
        cycle_count: int = 0,
        t_max: Optional[float] = None,
        dt: float = 60.0) -> Dict:
    """Run current-driven simulation (for sysdiagnose validation).

    i(t) is externally provided rather than solved from power.
    """
    if T_ambient is None:
        T_ambient = params.T_ambient
    if T_cell_init is None:
        T_cell_init = T_ambient
    if t_max is None:
        t_max = params.t_max_s

    N = params.N_RC
    y0 = np.zeros(N + 3)
    y0[0] = soc_init
    y0[1] = T_cell_init

    t_eval = np.arange(0.0, t_max, dt)

    def soc_empty_event(t, y, *_args):
        return y[0] - params.soc_empty
    soc_empty_event.terminal = True
    soc_empty_event.direction = -1

    min_tau = min((b.tau for b in params.rc_branches), default=1e6)
    max_step = min(60.0, min_tau / 2.0) if N > 0 else 60.0
    method = "Radau" if (N > 0 and min_tau < 20.0) else "RK45"

    sol = solve_ivp(
        fun=lambda t, y: battery_rhs_current_driven(
            t, y, current_fn, params, T_ambient, cycle_count),
        t_span=(0.0, t_max),
        y0=y0,
        method=method,
        t_eval=t_eval,
        events=soc_empty_event,
        max_step=max_step,
        rtol=1e-6,
        atol=1e-9,
    )

    t_out = sol.t
    soc_out = np.clip(sol.y[0], 0.0, 1.0)
    T_out = sol.y[1]
    i_R_out = [sol.y[2 + j] for j in range(N)]
    h_out = sol.y[2 + N]

    n_pts = len(t_out)
    voltage = np.zeros(n_pts)
    current_a = np.zeros(n_pts)

    for k in range(n_pts):
        i_a = current_fn(t_out[k])
        current_a[k] = i_a
        i_R_k = [i_R_out[j][k] for j in range(N)]
        voltage[k] = terminal_voltage(soc_out[k], i_a, T_out[k],
                                      i_R_k, h_out[k], params)

    tte = None
    if sol.t_events and len(sol.t_events[0]) > 0:
        tte = float(sol.t_events[0][0])

    return {
        "t": t_out,
        "soc": soc_out,
        "T_cell": T_out,
        "voltage": voltage,
        "current_ma": current_a * 1000.0,
        "current_input_a": current_a,
        "time_to_empty": tte,
        "i_R": i_R_out,
        "h": h_out,
        "solver_status": sol.status,
        "solver_message": sol.message,
    }


def time_to_empty_hours(result: Dict) -> float:
    """Extract TTE in hours from a simulation result dict."""
    if result["time_to_empty"] is not None:
        return result["time_to_empty"] / 3600.0
    # Extrapolate if simulation ended before SOC hit threshold
    soc = result["soc"]
    t = result["t"]
    if len(soc) > 1:
        dsoc = soc[0] - soc[-1]
        dt_total = t[-1] - t[0]
        if dsoc > 0 and dt_total > 0:
            rate = dsoc / dt_total  # SOC per second
            remaining = soc[-1] - DEFAULT.soc_empty
            return (dt_total + max(remaining, 0.0) / rate) / 3600.0
    return float("inf")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 10: Monte Carlo TTE and Tornado Sensitivity
# ═══════════════════════════════════════════════════════════════════════════════

MC_DEFAULT_SIGMA: Dict[str, float] = {
    "Q_design_mah": 0.05,              # +/-5% capacity
    "R0_ref": 0.20,                    # +/-20% resistance
    "overhead_screen_on_mw": 0.20,     # +/-20% screen-on overhead
    "overhead_screen_off_mw": 0.20,    # +/-20% screen-off overhead
    "screen_brightness_coeff_mw": 0.20, # +/-20% display power
    "alpha_Q": 0.30,                   # +/-30% temp sensitivity
    "hA": 0.30,                        # +/-30% thermal dissipation
}


def perturb_params(params: BatteryParams, rng: np.random.Generator,
                   rel_sigma: Optional[Dict[str, float]] = None
                   ) -> BatteryParams:
    """Create a perturbed copy of params for Monte Carlo sampling.

    Each named parameter is sampled from N(nominal, nominal*sigma),
    clamped to 10% of nominal as a floor.
    """
    if rel_sigma is None:
        rel_sigma = MC_DEFAULT_SIGMA

    new = deepcopy(params)
    for name, sigma in rel_sigma.items():
        current = getattr(new, name)
        perturbed = max(current * 0.1, rng.normal(current, current * sigma))
        setattr(new, name, perturbed)

    return new


def monte_carlo_tte(usage_fn: Callable[[float], Dict],
                    params: BatteryParams = DEFAULT,
                    N: int = 500,
                    cycle_count: int = 0,
                    seed: int = 42,
                    rel_sigma: Optional[Dict[str, float]] = None,
                    T_ambient: Optional[float] = None,
                    dt: float = 60.0) -> Dict:
    """Run Monte Carlo TTE analysis with parameter perturbation.

    Returns dict with: tte_hours (array), param_records (list of dicts),
    median, mean, std, ci_95_lo, ci_95_hi.
    """
    rng = np.random.default_rng(seed)
    ttes = np.zeros(N)
    param_records: List[Dict[str, float]] = []

    for k in range(N):
        p = perturb_params(params, rng, rel_sigma)
        result = simulate(usage_fn, p, cycle_count=cycle_count,
                          T_ambient=T_ambient, dt=dt)
        ttes[k] = time_to_empty_hours(result)
        record = {}
        for name in (rel_sigma or MC_DEFAULT_SIGMA):
            record[name] = getattr(p, name)
        param_records.append(record)

    return {
        "tte_hours": ttes,
        "param_records": param_records,
        "median": float(np.median(ttes)),
        "mean": float(np.mean(ttes)),
        "std": float(np.std(ttes)),
        "ci_95_lo": float(np.percentile(ttes, 2.5)),
        "ci_95_hi": float(np.percentile(ttes, 97.5)),
    }


def tornado_sensitivity(usage_fn: Callable[[float], Dict],
                        params: BatteryParams = DEFAULT,
                        cycle_count: int = 0,
                        T_ambient: Optional[float] = None,
                        rel_delta: Optional[Dict[str, float]] = None,
                        dt: float = 60.0
                        ) -> List[Tuple[str, float, float, float]]:
    """Parameter-wise TTE sensitivity for tornado chart.

    For each parameter, computes TTE at (nominal - delta) and (nominal + delta).

    Returns list of (param_name, tte_low, tte_nominal, tte_high) tuples,
    sorted by |swing| descending.
    """
    if rel_delta is None:
        rel_delta = MC_DEFAULT_SIGMA

    result_nom = simulate(usage_fn, params, cycle_count=cycle_count,
                          T_ambient=T_ambient, dt=dt)
    tte_nom = time_to_empty_hours(result_nom)

    sensitivities = []
    for name, delta_frac in rel_delta.items():
        nom_val = getattr(params, name)

        p_lo = deepcopy(params)
        setattr(p_lo, name, nom_val * (1.0 - delta_frac))
        result_lo = simulate(usage_fn, p_lo, cycle_count=cycle_count,
                             T_ambient=T_ambient, dt=dt)
        tte_lo = time_to_empty_hours(result_lo)

        p_hi = deepcopy(params)
        setattr(p_hi, name, nom_val * (1.0 + delta_frac))
        result_hi = simulate(usage_fn, p_hi, cycle_count=cycle_count,
                             T_ambient=T_ambient, dt=dt)
        tte_hi = time_to_empty_hours(result_hi)

        sensitivities.append((name, tte_lo, tte_nom, tte_hi))

    sensitivities.sort(key=lambda x: abs(x[3] - x[1]), reverse=True)
    return sensitivities


# ═══════════════════════════════════════════════════════════════════════════════
# Section 11: Usage Schedule Builders
# ═══════════════════════════════════════════════════════════════════════════════

def make_constant_usage(usage_state: Dict) -> Callable[[float], Dict]:
    """Constant usage over time."""
    def schedule(t: float) -> Dict:
        return usage_state
    return schedule


def make_time_varying_usage(
        segments: List[Tuple[float, Dict]]) -> Callable[[float], Dict]:
    """Piecewise-constant usage from (duration_s, usage_dict) segments."""
    boundaries = [0.0]
    for dur, _ in segments:
        boundaries.append(boundaries[-1] + dur)

    def schedule(t: float) -> Dict:
        for idx, (dur, state) in enumerate(segments):
            if t < boundaries[idx + 1]:
                return state
        return segments[-1][1]

    return schedule


def make_daily_usage(wake_hour: int = 7, sleep_hour: int = 23,
                     active_fraction: float = 0.4,
                     brightness: float = 0.5,
                     wifi: bool = True, cellular: bool = True
                     ) -> Callable[[float], Dict]:
    """Realistic daily usage with sleep/wake cycling and intermittent use."""
    wake_s = wake_hour * 3600
    sleep_s = sleep_hour * 3600
    block_s = 300  # 5-minute blocks

    sleeping: Dict = {
        "screen_on": False, "brightness": 0.0, "cpu_load": 0.0,
        "wifi_state": "idle" if wifi else "off", "wifi_data_rate": 0.0,
        "cellular_state": "idle" if cellular else "off",
        "cellular_signal": 0.8, "cellular_data_rate": 0.0,
        "gps_state": "off", "bluetooth_state": "idle",
        "audio_state": "off", "camera_state": "off",
    }
    active: Dict = {
        "screen_on": True, "brightness": brightness, "cpu_load": 0.15,
        "wifi_state": "active" if wifi else "off", "wifi_data_rate": 0.2,
        "cellular_state": "idle" if wifi else "active",
        "cellular_signal": 0.8,
        "cellular_data_rate": 0.0 if wifi else 0.2,
        "gps_state": "off", "bluetooth_state": "idle",
        "audio_state": "off", "camera_state": "off",
    }
    idle_awake: Dict = {
        "screen_on": False, "brightness": 0.0, "cpu_load": 0.02,
        "wifi_state": "idle" if wifi else "off", "wifi_data_rate": 0.0,
        "cellular_state": "idle" if cellular else "off",
        "cellular_signal": 0.8, "cellular_data_rate": 0.0,
        "gps_state": "off", "bluetooth_state": "idle",
        "audio_state": "off", "camera_state": "off",
    }

    def schedule(t: float) -> Dict:
        t_mod = t % 86400
        if t_mod < wake_s or t_mod >= sleep_s:
            return sleeping
        block_idx = int((t_mod - wake_s) / block_s)
        if (block_idx % 10) / 10.0 < active_fraction:
            return active
        return idle_awake

    return schedule


# ═══════════════════════════════════════════════════════════════════════════════
# Section 12: Predefined Scenarios
# ═══════════════════════════════════════════════════════════════════════════════

SCENARIOS: Dict[str, Dict] = {
    "idle_screen_off": {
        "screen_on": False, "brightness": 0.0, "cpu_load": 0.0,
        "wifi_state": "idle", "wifi_data_rate": 0.0,
        "cellular_state": "idle", "cellular_signal": 0.8,
        "cellular_data_rate": 0.0,
        "gps_state": "off", "bluetooth_state": "off",
        "audio_state": "off", "camera_state": "off",
    },
    "light_browsing": {
        "screen_on": True, "brightness": 0.5, "cpu_load": 0.1,
        "wifi_state": "active", "wifi_data_rate": 0.1,
        "cellular_state": "idle", "cellular_signal": 0.8,
        "cellular_data_rate": 0.0,
        "gps_state": "off", "bluetooth_state": "off",
        "audio_state": "off", "camera_state": "off",
    },
    "video_streaming": {
        "screen_on": True, "brightness": 0.7, "cpu_load": 0.3,
        "wifi_state": "active", "wifi_data_rate": 0.5,
        "cellular_state": "idle", "cellular_signal": 0.8,
        "cellular_data_rate": 0.0,
        "gps_state": "off", "bluetooth_state": "off",
        "audio_state": "speaker", "camera_state": "off",
    },
    "navigation": {
        "screen_on": True, "brightness": 0.8, "cpu_load": 0.25,
        "wifi_state": "off", "wifi_data_rate": 0.0,
        "cellular_state": "active", "cellular_signal": 0.6,
        "cellular_data_rate": 0.3,
        "gps_state": "active", "bluetooth_state": "off",
        "audio_state": "speaker", "camera_state": "off",
    },
    "gaming": {
        "screen_on": True, "brightness": 0.8, "cpu_load": 0.8,
        "wifi_state": "active", "wifi_data_rate": 0.3,
        "cellular_state": "idle", "cellular_signal": 0.8,
        "cellular_data_rate": 0.0,
        "gps_state": "off", "bluetooth_state": "off",
        "audio_state": "speaker", "camera_state": "off",
    },
    "video_call": {
        "screen_on": True, "brightness": 0.6, "cpu_load": 0.4,
        "wifi_state": "active", "wifi_data_rate": 0.4,
        "cellular_state": "idle", "cellular_signal": 0.8,
        "cellular_data_rate": 0.0,
        "gps_state": "off", "bluetooth_state": "off",
        "audio_state": "speaker", "camera_state": "viewfinder",
    },
    "music_screen_off": {
        "screen_on": False, "brightness": 0.0, "cpu_load": 0.05,
        "wifi_state": "active", "wifi_data_rate": 0.05,
        "cellular_state": "idle", "cellular_signal": 0.8,
        "cellular_data_rate": 0.0,
        "gps_state": "off", "bluetooth_state": "active",
        "audio_state": "headphone", "camera_state": "off",
    },
    "poor_signal_idle": {
        "screen_on": False, "brightness": 0.0, "cpu_load": 0.0,
        "wifi_state": "off", "wifi_data_rate": 0.0,
        "cellular_state": "idle", "cellular_signal": 0.2,
        "cellular_data_rate": 0.0,
        "gps_state": "off", "bluetooth_state": "off",
        "audio_state": "off", "camera_state": "off",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Section 13: Self-Test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("battery_model.py -- Paper-exact self-test")
    print("=" * 60)

    # Parameter summary
    p = DEFAULT
    print(f"Q_design  = {p.Q_design_mah} mAh")
    print(f"R0_ref    = {p.R0_ref} Ohm")
    print(f"N_RC      = {p.N_RC}, state_dim = {p.state_dim}")
    for j, b in enumerate(p.rc_branches):
        print(f"  RC[{j}]: R={b.R_ref:.3f} Ohm, C={b.C:.0f} F, "
              f"tau={b.tau:.1f} s, alpha_R={b.alpha_R}")
    print(f"Overhead: screen_on={p.overhead_screen_on_mw} mW, "
          f"screen_off={p.overhead_screen_off_mw} mW")
    print(f"Thermal:  C_th={p.C_th}, hA={p.hA}, "
          f"alpha_R0={p.alpha_R0}, alpha_Q={p.alpha_Q}")
    print(f"Aging:    alpha_fade={p.alpha_fade} "
          f"{'(DISABLED -- not in paper)' if p.alpha_fade == 0 else ''}")
    print(f"Hyst:     gamma={p.gamma}, M0={p.M0}, M1={p.M1}")
    print()

    # Thermal spot checks (Eqs 3-5)
    print("Thermal spot checks (Eqs 3-5):")
    print(f"  {'T (C)':>7s}  {'R0 (Ohm)':>10s}  {'Q_eff (mAh)':>12s}")
    for T_test in [25.0, 15.0, 5.0, 0.0, -10.0]:
        r = R0_at_temp(0.159, T_test)
        q = Q_eff_at_temp(4329.0, T_test)
        print(f"  {T_test:7.1f}  {r:10.4f}  {q:12.0f}")
    print()

    # Scenario TTEs (paper-default: no aging, full design capacity)
    print("Scenario TTE (paper-default, no aging, 25C):")
    print(f"  {'Scenario':25s}  {'Power (mW)':>10s}  {'TTE (h)':>8s}")
    print("-" * 50)
    for name in ["idle_screen_off", "light_browsing", "video_streaming",
                 "navigation", "gaming", "music_screen_off"]:
        usage = make_constant_usage(SCENARIOS[name])
        result = simulate(usage, dt=60.0)
        tte = time_to_empty_hours(result)
        pw = total_power(SCENARIOS[name])
        print(f"  {name:25s}  {pw:10.0f}  {tte:8.1f}")
    print()

    # Quick daily usage test
    print("Daily usage TTE (paper-default):")
    daily = make_daily_usage(active_fraction=0.4, brightness=0.5)
    result = simulate(daily, dt=60.0)
    tte = time_to_empty_hours(result)
    print(f"  40% active, 50% brightness -> {tte:.1f} h")
    print()

    # Optional: show aging extension
    print("--- Optional aging extension (not in paper) ---")
    p_aged = BatteryParams(alpha_fade=0.005232, beta_fade=0.5)
    for nc in [0, 100, 500, 716, 1000]:
        cap = capacity_after_cycles(nc, p_aged)
        pct = cap / p_aged.Q_design_mah * 100
        print(f"  {nc:4d} cycles -> {cap:.0f} mAh ({pct:.1f}%)")
    print()

    print("Self-test complete.")
